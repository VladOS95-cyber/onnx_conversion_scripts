# !pip install --upgrade torch==2.6.0 torchaudio==2.6.0 numpy==2.2.6 librosa==0.11.0 onnx==1.18.0 onnxslim==0.1.59

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio as ta
from torchaudio.compliance.kaldi import get_mel_banks

from s3tokenizer.model import Conv1d, LayerNorm, Linear, MultiHeadAttention
from s3tokenizer.utils import mask_to_bias

import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import math
from tqdm import tqdm


# Sampling rate of the inputs to S3TokenizerV2
S3GEN_SR = 24000
S3_SR = 16_000
S3_HOP = 160
S3_TOKEN_HOP = 640
S3_TOKEN_RATE = 25 # 25 tokens/sec
SPEECH_VOCAB_SIZE = 6561
MILLISECONDS_TO_SECONDS = 0.001

START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
EXAGGERATION_TOKEN = 6563

ENC_COND_LEN = 6 * S3_SR
DEC_COND_LEN = 10 * S3GEN_SR

CFM_PARAMS = {
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
}
ISTFT_PARAMS = {"n_fft": 16, "hop_len": 4}

# override certain torch functions
torch.Tensor.item = lambda x: x # no-op


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8

    use_sdpa: bool = False

def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    1 for non-padded part and 0 for padded part.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    # max_len = max_len if max_len > 0 else lengths.max()
    max_len_2 = lengths.max()
    seq_range = torch.arange(0,
                             max_len_2,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len_2)
    seq_length_expand = lengths.unsqueeze(-1)
    return seq_range_expand < seq_length_expand


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         scaling=None):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    if scaling is not None:
        t = t * scaling
    freqs = torch.outer(t, freqs).float()  # type: ignore
    
    cos = freqs.cos()
    sin = freqs.sin()
    
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, :D // 2], xq[:, :, :, D // 2:]
    xq_r = torch.cat((-half_r, half_l), dim=-1)

    D = xk.shape[-1]

    half_l, half_r = xk[:, :, :, :D // 2], xk[:, :, :, D // 2:]
    xk_r = torch.cat((-half_r, half_l), dim=-1)

    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


class FSQCodebook(nn.Module):

    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    @torch.inference_mode()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x = rearrange(x, "... d -> (...) d")
        x = x.reshape(-1, x.shape[-1])
        return x

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        h = self.project_down(x).float()
        h = h.tanh()
        h = h * 0.9990000128746033
        h = h.round() + 1
        # h = ((self.level - 1) * h).round()  # range [-k, k]
        powers = torch.pow(
            self.level,
            torch.arange(2**self.level, device=x.device, dtype=h.dtype))
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)
        ind = mu.reshape(x_shape[0], x_shape[1])
        return ind

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'There is no official up project component provided')


class FSQVectorQuantization(nn.Module):
    """Vector quantization implementation (inference-only).
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
    ):
        super().__init__()
        assert 3**8 == codebook_size
        self._codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(x)

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        quantize = self._codebook.decode(embed_ind)
        # quantize = rearrange(quantize, "b n d -> b d n")
        quantize = quantize.permute(0, 2, 1)
        return quantize


class FSMNMultiHeadAttention(MultiHeadAttention):

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__(n_state, n_head)

        self.fsmn_block = nn.Conv1d(n_state,
                                          n_state,
                                          kernel_size,
                                          stride=1,
                                          padding=0,
                                          groups=n_state,
                                          bias=False)
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = nn.ConstantPad1d(
            (self.left_padding, self.right_padding), 0.0)

        self.use_sdpa = use_sdpa

    def forward_fsmn(self,
                     inputs: torch.Tensor,
                     mask: Optional[torch.Tensor] = None):
        b, t, _, _ = inputs.size()
        inputs = inputs.view(b, t, -1)
        if mask is not None and mask.size(2) > 0:  # time2 > 0
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        return x * mask

    def qkv_attention(self,
                      q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      mask_pad: Optional[torch.Tensor] = None,
                      cos: Optional[torch.Tensor] = None,
                      sin: Optional[torch.Tensor] = None):
        _, _, D = q.shape
        scale = (D // self.n_head)**-0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if cos is not None and sin is not None:
            q, k = apply_rotary_emb(q, k, cos=cos, sin=sin)

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k  # (B, n_head, T, T)
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = F.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(
                0, 2, 1, 3).flatten(start_dim=2), qk.detach(), fsm_memory
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.,
                scale=1.,
            )
            output = (output.transpose(1,
                                       2).contiguous().view(q.size(0), -1, D)
                      )  # (batch, time1, d_model)
            return output, None, fsm_memory

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_pad: Optional[torch.Tensor] = None,
                cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad,
                                                cos, sin)
        return self.out(wv) + fsm_memory, qk


class ResidualAttentionBlock(nn.Module):

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__()

        self.attn = FSMNMultiHeadAttention(n_state,
                                           n_head,
                                           kernel_size,
                                           use_sdpa=use_sdpa)
        self.attn_ln = LayerNorm(n_state, eps=1e-6)

        n_mlp = n_state * 4

        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(),
                                       Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(
            self.attn_ln(x), mask=mask, mask_pad=mask_pad,
            cos=cos, sin=sin)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = Conv1d(n_mels,
                            n_state,
                            kernel_size=3,
                            stride=stride,
                            padding=1)
        self.conv2 = Conv1d(n_state,
                            n_state,
                            kernel_size=3,
                            stride=2,
                            padding=1)
        cos, sin = precompute_freqs_cis(64, 1024 * 2)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])

    def forward(self, x: torch.Tensor,
                x_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = F.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = F.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        # NOTE: .contiguous() is essential for dynamo export!
        x = x.contiguous()

        
        cos = self.cos[:x.size(1)].to(x.device)
        sin = self.sin[:x.size(1)].to(x.device)

        mask_pad = mask.transpose(1, 2)
        mask = mask_to_bias(mask, x.dtype).unsqueeze(1)

        for block in self.blocks:
            x = block(x, mask, mask_pad, cos, sin)

        return x, x_len


class S3TokenizerV2(nn.Module):
    """S3 tokenizer v2 implementation (inference-only).
    Args:
        config (ModelConfig): Config
    """

    def __init__(self):
        super().__init__()
        # self.name = name  # Store model name for token_rate determination
        self.config = ModelConfig()
        self.encoder = AudioEncoderV2(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,
            self.config.use_sdpa,
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

    def forward(self, mel: torch.Tensor,
                mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: torch.Tensor,
                 mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize mel spectrogram to tokens, with automatic long audio handling.

        Args:
            mel: mel spectrogram tensor, shape (batch_size, n_mels, T)
            mel_len: mel length tensor, shape (batch_size,)

        Returns:
            code: quantized tokens, shape (batch_size, T')
            code_len: token length, shape (batch_size,)
        """
        # Check if any audio in the batch exceeds 30 seconds
        # Assuming 16kHz sample rate and hop_length=160, 30s = 30*16000/160 = 3000 frames
        # max_frames = 3000

        # Check which samples are long audio
        # assert (mel_len <= max_frames).all()

        # All short audio - use original method
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden).long()
        return code, code_len

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False


class S3Tokenizer(S3TokenizerV2):
    """
    s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        config: ModelConfig = ModelConfig()
    ):
        super().__init__()

        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        max_len,
    ) -> torch.Tensor:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).

        Args
        ----
        - `wavs`: 16 kHz speech audio
        """
        mels = self.log_mel_spectrogram(wavs)  # [B, F, T]
        if max_len is not None:
            mels = mels[..., :max_len * 4]
        mel_lens = torch.full((mels.shape[0],), mels.shape[-1], dtype=torch.int32, device=self.device)

        speech_tokens, _ = self.quantize(mels, mel_lens)
        return speech_tokens

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (128, n_frames)
            A Tensor that contains the Mel spectrogram
        """

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=False
        )
        # remove Nyquist bin
        stft = stft[..., :-1, :]
        # compute magnitude squared
        magnitudes = stft[..., 0]**2 + stft[..., 1]**2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


class SafeDenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(SafeDenseLayer, self).__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = torch.nn.Sequential()
        self.nonlinear.add_module("layernorm", torch.nn.LayerNorm(out_channels))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.linear(x)
        if x.size(-1) == 1:
            x = x[:, :, 0]
        x = self.nonlinear(x)
        return x


class PrepareConditionalsModel(torch.nn.Module):

    speech_cond_prompt_len = 150
    speaker_embed_size = 256

    def __init__(self, chatterbox):
        super().__init__()

        # TODO: Move loading elsewhere
        self.s3 = S3Tokenizer()
        self.s3.load_state_dict(chatterbox.s3gen.tokenizer.state_dict(), strict=False)

        self.speaker_encoder = chatterbox.s3gen.speaker_encoder
        self.flow = chatterbox.s3gen.flow

        self.cond_enc = chatterbox.t3.cond_enc

        self.resampler = ta.transforms.Resample(S3GEN_SR, S3_SR)
        self.eps = torch.tensor(torch.finfo(torch.float).eps)
        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=128
        )
        self.register_buffer(
            "mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

        self.speech_emb = chatterbox.t3.speech_emb
        self.speech_pos_emb = chatterbox.t3.speech_pos_emb

        # Speaker embedding projection
        # NOTE: From testing, randomly/zero initializing speaker embedding seems to work fine
        # speaker_emb = torch.randn(batch_size, self.speaker_embed_size)
        speaker_emb = torch.zeros(1, self.speaker_embed_size)
        self.cond_spkr = self.cond_enc.spkr_enc(speaker_emb.view(-1, self.speaker_embed_size))[:, None]  # (B, 1, dim)

    def mel_spectrogram(self, y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
        y = F.pad(
            y.unsqueeze(1),
            ((n_fft - hop_size) // 2, (n_fft - hop_size) // 2),
            mode="reflect",
        )
        y = y.squeeze(1)
        hann_window = torch.hann_window(win_size)
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel = torch.from_numpy(mel).float()
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        # real = spec[..., 0]
        # imag = spec[..., 1]
        # spec = torch.sqrt(real**2 + imag**2 + 1e-9)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(mel, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1) # spectral_normalize_torch

        return spec
    
    def _next_power_of_2(self, x: int) -> int:
        r"""Returns the smallest power of 2 that is greater than x"""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    
    def _get_strided(self, waveform: torch.Tensor, window_size: int, window_shift: int) -> torch.Tensor:
        r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
        representing how the window is shifted along the waveform. Each row is a frame.

        Args:
            waveform (Tensor): Tensor of size ``num_samples``
            window_size (int): Frame length
            window_shift (int): Frame shift
            snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
                in the file, and the number of frames depends on the frame_length.  If False, the number of frames
                depends only on the frame_shift, and we reflect the data at the ends.

        Returns:
            Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
        """
        num_samples = waveform.size(0)
        strides = (window_shift * waveform.stride(0), waveform.stride(0))

        if num_samples < window_size:
            return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift

        sizes = (m, window_size)
        return waveform.as_strided(sizes, strides)

    def _get_log_energy(self, strided_input: torch.Tensor, epsilon: torch.Tensor, energy_floor: float) -> torch.Tensor:
        r"""Returns the log energy of size (m) for a strided_input (m,*)"""
        device, dtype = strided_input.device, strided_input.dtype
        log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
        if energy_floor == 0.0:
            return log_energy
        return torch.max(log_energy, torch.tensor(math.log(energy_floor), device=device, dtype=dtype))

    def _get_window(
        self,
        waveform: torch.Tensor,
        padded_window_size: int,
        window_size: int,
        window_shift: int,
        preemphasis_coefficient: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Gets a window and its log energy

        Returns:
            (Tensor, Tensor): strided_input of size (m, ``padded_window_size``) and signal_log_energy of size (m)
        """
        device, dtype = waveform.device, waveform.dtype
        # size (m, window_size)
        strided_input = self._get_strided(waveform, window_size, window_shift)

        # Subtract each row/frame by its mean
        row_means = torch.mean(strided_input, dim=1).unsqueeze(1)  # size (m, 1)
        strided_input = strided_input - row_means

        if preemphasis_coefficient != 0.0:
            # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
            offset_strided_input = F.pad(strided_input.unsqueeze(0), (1, 0), mode="replicate").squeeze(
                0
            )  # size (m, window_size + 1)
            strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

        # Apply window_function to each row/frame
        window_function = torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85).unsqueeze(0)  # size (1, window_size)
        strided_input = strided_input * window_function  # size (m, window_size)

        # Pad columns with zero until we reach size (m, padded_window_size)
        if padded_window_size != window_size:
            padding_right = padded_window_size - window_size
            strided_input = F.pad(
                strided_input.unsqueeze(0), (0, padding_right), mode="constant", value=0
            ).squeeze(0)

        return strided_input

    def _get_waveform_and_window_properties(
        self,
        waveform: torch.Tensor,
        channel: int,
        sample_frequency: float,
        frame_shift: float,
        frame_length: float,
    ) -> Tuple[torch.Tensor, int, int, int]:
        r"""Gets the waveform and window properties"""
        channel = max(channel, 0)
        waveform = waveform[channel, :]  # size (n)
        window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
        window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
        padded_window_size = self._next_power_of_2(window_size)
        return waveform, window_shift, window_size, padded_window_size

    def extract_feature(self, waveform: torch.Tensor,
        channel: int = -1,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        high_freq: float = 0.0,
        low_freq: float = 20.0,
        num_mel_bins: int = 23,
        preemphasis_coefficient: float = 0.97,
        sample_frequency: float = 16000.0,
        vtln_high: float = -500.0,
        vtln_low: float = 100.0,
        vtln_warp: float = 1.0):

        device, dtype = waveform.device, waveform.dtype
        waveform, window_shift, window_size, padded_window_size = self._get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length)

        # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
        strided_input = self._get_window(
            waveform,
            padded_window_size,
            window_size,
            window_shift,
            preemphasis_coefficient,
        )

        # size (m, padded_window_size // 2 + 1)
        spec = torch.stft(
            strided_input,
            n_fft=512,
            hop_length=512,
            center=False,
            window=None,
            return_complex=False
        )   # shape: [..., freq, 2]  (last dim = [real, imag])

        # Compute magnitude manually
        real = spec[..., 0]
        imag = spec[..., 1]
        spectrum = torch.sqrt(real**2 + imag**2).squeeze(-1)
        spectrum = spectrum.pow(2.0)

        # size (num_mel_bins, padded_window_size // 2)
        mel_energies, _ = get_mel_banks(
            num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq, vtln_low, vtln_high, vtln_warp
        )
        mel_energies = mel_energies.to(device=device, dtype=dtype)

        # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = F.pad(mel_energies, (0, 1), mode="constant", value=0)

        # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
        mel_energies = torch.matmul(spectrum, mel_energies.T)

        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = torch.max(mel_energies, self.eps).log()
        return mel_energies

    def prepare_conditions_from_audio(self, audio_values):
        batch_size = audio_values.shape[0]

        # Compute embed_ref
        ref_wav_24 = audio_values[..., :DEC_COND_LEN]
        speaker_features = self.mel_spectrogram(ref_wav_24).transpose(1, 2)

        # Resample to 16kHz
        ref_wav_16 = self.resampler(audio_values) # resample uncropped audio

        # Speech cond prompt tokens
        # TODO START REMOVE
        # -- AT EXPORT, WE MUST SWAP THIS WITH self.resampler(audio_values)
        # ref_wav_16 = librosa.resample(audio_values.cpu().numpy(), orig_sr=S3GEN_SR, target_sr=S3_SR)
        # ref_wav_16 = torch.from_numpy(ref_wav_16).to(audio_values.device)
        # TODO END REMOVE

        feature = self.extract_feature(ref_wav_16, num_mel_bins=80) # == Kaldi.fbank(ref_wav_16, num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        speaker_embeddings = self.speaker_encoder(feature.unsqueeze(0))

        t3_cond_prompt_tokens = self.s3(ref_wav_16[..., :ENC_COND_LEN], max_len=self.speech_cond_prompt_len)

        resampled_wav_16 = self.resampler(ref_wav_24) # resample uncropped audio

        # NOTE: For some reason, we do two passes of the s3 tokenizer
        # TODO: Try reduce this?
        # Tokenize 16khz reference
        prompt_token = self.s3(resampled_wav_16, max_len=None)

        cond_prompt_speech_emb = self.speech_emb(t3_cond_prompt_tokens) + \
                     self.speech_pos_emb(t3_cond_prompt_tokens)

        # Cond prompt
        cond_prompt_speech_emb = self.cond_enc.perceiver(cond_prompt_speech_emb)

        expanded_cond_spkr = self.cond_spkr.expand(batch_size, -1, -1)  # (B, 1, dim)

        # Concat and return
        cond_emb = torch.cat((
            expanded_cond_spkr,
            cond_prompt_speech_emb,
        ), dim=1)  # (B, len_cond, dim)
        # assert cond_emb.dim() == 3
        return cond_emb, prompt_token, speaker_embeddings, speaker_features

    def forward(
        self,
        audio_values: torch.Tensor, # NOTE: Must have sample rate of S3GEN_SR=24000
    ):
        cond_emb, prompt_token, speaker_embeddings, speaker_features = self.prepare_conditions_from_audio(audio_values)
        return cond_emb, prompt_token, speaker_embeddings, speaker_features


class InputsEmbeds(nn.Module):
    def __init__(self, chatterbox):
        super().__init__()
        self.text_emb = chatterbox.t3.text_emb
        self.text_pos_emb = chatterbox.t3.text_pos_emb.emb

        self.speech_emb = chatterbox.t3.speech_emb
        self.speech_pos_emb = chatterbox.t3.speech_pos_emb.emb

        self.emotion_adv_fc = chatterbox.t3.cond_enc.emotion_adv_fc

    def forward(self, input_ids, position_ids, exaggeration):
        assert position_ids.shape == input_ids.shape
        batch_size, seq_len = input_ids.shape

        x = input_ids
        idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Detect first zero
        is_zero = (x == 0)
        has_zero = is_zero.any(dim=1)
        zero_pos = torch.where(
            has_zero,
            is_zero.float().argmax(dim=1),
            torch.full((batch_size,), -1, device=x.device)  # placeholder
        )

        # Masks
        exaggeration_mask = (x == EXAGGERATION_TOKEN)
        base_text_mask = (idx <= zero_pos.unsqueeze(1)) & has_zero.unsqueeze(1)
        
        text_mask = base_text_mask & ~exaggeration_mask
        speech_mask = ~base_text_mask & ~exaggeration_mask

        # Compute relative positions by multiplying with the masks
        text_pos_ids = position_ids * text_mask
        speech_pos_ids = position_ids * speech_mask

        # Flatten
        flat_x = x.view(-1)
        flat_text_mask = text_mask.view(-1)
        flat_speech_mask = speech_mask.view(-1)
        flat_exaggeration_mask = exaggeration_mask.view(-1)
        flat_text_pos = text_pos_ids.view(-1)
        flat_speech_pos = speech_pos_ids.view(-1)

        # Replace invalid indices with 0 (safe padding idx)
        safe_text_idx = torch.where(flat_text_mask, flat_x, torch.zeros_like(flat_x))
        safe_text_pos = torch.where(flat_text_mask, flat_text_pos, torch.zeros_like(flat_text_pos))

        safe_speech_idx = torch.where(flat_speech_mask, flat_x, torch.zeros_like(flat_x))
        safe_speech_pos = torch.where(flat_speech_mask, flat_speech_pos, torch.zeros_like(flat_speech_pos))

        # Embed everything, but irrelevant positions will become "padding" embeddings
        all_text_emb = self.text_emb(safe_text_idx) + self.text_pos_emb(safe_text_pos)
        all_speech_emb = self.speech_emb(safe_speech_idx) + self.speech_pos_emb(safe_speech_pos)

        # Finally, mask out the padding positions to zero them
        text_emb = all_text_emb * flat_text_mask.unsqueeze(-1)
        speech_emb = all_speech_emb * flat_speech_mask.unsqueeze(-1)

        # Emotion Adv: must provide a value if this model uses emotion conditioning
        emotion_adv = exaggeration.view(-1, 1, 1)
        cond_emotion_adv = self.emotion_adv_fc(emotion_adv)

        # Reshape to [B*L, D] to match masks
        embed_dim = text_emb.size(-1)
        text_emb_full   = text_emb
        speech_emb_full = speech_emb

        # Start with zeros
        out = torch.zeros(batch_size * seq_len, embed_dim, device=x.device, dtype=text_emb.dtype)

        # Where text mask is True → take text_emb, else keep current out
        out = torch.where(flat_text_mask.unsqueeze(-1), text_emb_full, out)

        # Where speech mask is True → take speech_emb, else keep current out
        out = torch.where(flat_speech_mask.unsqueeze(-1), speech_emb_full, out)
        
        # Handle exaggeration tokens
        # We need to expand cond_emotion_adv to match the number of exaggeration tokens
        # This assumes cond_emotion_adv is (batch_size, 1, dim) and we need to map it correctly
        # to the flattened positions.
        # We can create an index mapping from the flattened index to the batch index.
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len).reshape(-1)
        exaggeration_emb_full = cond_emotion_adv[batch_indices].transpose(0, 1) 

        # Zero out positions where mask is False
        exaggeration_emb = exaggeration_emb_full * flat_exaggeration_mask.unsqueeze(-1)

        out = out + exaggeration_emb
        out = out.view(batch_size, seq_len, embed_dim)
        return out


class ISTFT(torch.nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        assert n_fft >= win_length
        super().__init__()

        self.filter_length = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = self.filter_length // 2 + 1
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])


        self.window = torch.hann_window(win_length)

        # Center pad the window to the size of n_fft
        pad_length = n_fft - self.window.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left

        torch_fft_window = F.pad(self.window, (pad_left, pad_right), mode='constant', value=0)
        inverse_basis *= torch_fft_window

        self.register_buffer('inverse_basis', inverse_basis.float())

    @staticmethod
    def window_sumsquare(
        window,
        n_frames,
        hop_length,
        win_length,
        n_fft,
    ):
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)

        # Compute the squared window at the desired length
        win_sq = window ** 2

        # Center pad the window to the size of n_fft
        pad_length = n_fft - win_sq.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        win_sq = F.pad(win_sq, (pad_left, pad_right), mode='constant', value=0)

        # Prepare the kernel for conv_transpose1d
        win_sq = win_sq.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_fft)

        # Create the input signal: ones of shape (1, 1, n_frames)
        s = torch.ones(1, 1, n_frames, dtype=window.dtype, device=window.device)

        # Perform conv_transpose1d with stride=hop_length
        x = F.conv_transpose1d(s, win_sq, stride=hop_length).squeeze()

        # Adjust x to have length n
        x = x[:n]

        return x

    def forward(self, recombine_magnitude_phase):
        assert recombine_magnitude_phase.dim() == 3, 'must be [B, 2 * N, T]'
        num_frames = recombine_magnitude_phase.size(-1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        window_sum = self.window_sumsquare(
            self.window,
            n_frames=num_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.filter_length,
        )

        tiny_value = torch.finfo(window_sum.dtype).tiny

        denom = torch.where(
            window_sum > tiny_value,
            window_sum,
            torch.tensor(1.0, dtype=window_sum.dtype, device=window_sum.device),
        )
        # Apply the transformation
        inverse_transform /= denom

        # scale by hop ratio
        inverse_transform *= self.filter_length / self.hop_length

        q = self.filter_length // 2
        inverse_transform = inverse_transform[:, 0, q:-q]
        return inverse_transform

istft = ISTFT(ISTFT_PARAMS["n_fft"], ISTFT_PARAMS["hop_len"], ISTFT_PARAMS["n_fft"])


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max()
    seq_range = torch.arange(0,
                            max_len,
                            dtype=torch.int64,
                            device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e+10
    return mask


class ConditionalDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.output_size = model.s3gen.flow.output_size
        self.input_embedding = model.s3gen.flow.input_embedding
        self.spk_embed_affine_layer = model.s3gen.flow.spk_embed_affine_layer
        self.encoder = model.s3gen.flow.encoder
        self.encoder_proj = model.s3gen.flow.encoder_proj
        self.time_embeddings = model.s3gen.flow.decoder.estimator.time_embeddings
        self.time_mlp = model.s3gen.flow.decoder.estimator.time_mlp
        self.up_blocks = model.s3gen.flow.decoder.estimator.up_blocks
        self.static_chunk_size = model.s3gen.flow.decoder.estimator.static_chunk_size
        self.mid_blocks = model.s3gen.flow.decoder.estimator.mid_blocks
        self.down_blocks = model.s3gen.flow.decoder.estimator.down_blocks
        self.final_block = model.s3gen.flow.decoder.estimator.final_block
        self.final_proj = model.s3gen.flow.decoder.estimator.final_proj
        self.n_fft = ISTFT_PARAMS["n_fft"]
        self.hop_len = ISTFT_PARAMS["hop_len"]
        self.n_trim = S3GEN_SR // 50
        self.stft_window = model.s3gen.mel2wav.stft_window
        self.f0_predictor = model.s3gen.mel2wav.f0_predictor
        self.f0_upsamp = model.s3gen.mel2wav.f0_upsamp
        self.m_source = model.s3gen.mel2wav.m_source
        self.inference_cfg_rate = 0.7
        self.conv_pre = model.s3gen.mel2wav.conv_pre
        self.lrelu_slope = model.s3gen.mel2wav.lrelu_slope
        self.reflection_pad = model.s3gen.mel2wav.reflection_pad
        self.ups = model.s3gen.mel2wav.ups
        self.source_downs = model.s3gen.mel2wav.source_downs
        self.source_resblocks = model.s3gen.mel2wav.source_resblocks
        self.resblocks = model.s3gen.mel2wav.resblocks
        self.conv_post = model.s3gen.mel2wav.conv_post
        self.istft = istft
    
    def cond_forward(self, x, mask, mu, t, spks, cond) -> torch.Tensor:
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.
        """

        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = torch.cat([x, mu], dim=1)
        spks = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, spks], dim=1)
        x = torch.cat([x, cond], dim=1)

        masks = [mask]
        resnet, transformer_blocks, downsample = self.down_blocks[0]
        mask_down = masks[-1]
        x = resnet(x, mask_down, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = mask_to_bias(mask_down.bool() == 1, x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(
                hidden_states=x,
                attention_mask=attn_mask,
                timestep=t,
            )
        x = x.permute(0, 2, 1).contiguous()
        residual = x  # Save hidden states for skip connections
        x = downsample(x * mask_down)
        masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = x.permute(0, 2, 1).contiguous()
            attn_mask = mask_to_bias(mask_mid.bool() == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = x.permute(0, 2, 1).contiguous() 

        resnet, transformer_blocks, upsample = self.up_blocks[0]
        mask_up = masks.pop()
        x = torch.cat([x[:, :, :residual.shape[-1]], residual], dim=1)
        x = resnet(x, mask_up, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = mask_to_bias(mask_up.bool() == 1, x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(
                hidden_states=x,
                attention_mask=attn_mask,
                timestep=t,
            )
        x = x.permute(0, 2, 1).contiguous()
        x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output

    def flow_forward(self, speech_tokens, token_len, mask, embedding, prompt_feat):
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        speech_tokens = self.input_embedding(torch.clamp(speech_tokens, min=0))
        speech_tokens = speech_tokens * mask

        # text encode
        text_encoded, _ = self.encoder(speech_tokens, token_len)
        mel_len1, mel_len2 = prompt_feat.shape[1], text_encoded.shape[1] - prompt_feat.shape[1]
        text_encoded = self.encoder_proj(text_encoded)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size]).to(text_encoded.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mu = text_encoded
        spks = embedding
        if not isinstance(mel_len1, torch.Tensor):
            mel_len1 = torch.tensor(mel_len1, device=speech_tokens.device)
        if not isinstance(mel_len2, torch.Tensor):
            mel_len2 = torch.tensor(mel_len2, device=speech_tokens.device)
        return mel_len1, mel_len2, mu, spks, conds

    def decode(self, x: torch.Tensor, s_stft: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)

        # ---- Upsample 0 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[0](x)

        si = self.source_downs[0](s_stft)
        si = self.source_resblocks[0](si)
        x = x + si

        xs0 = self.resblocks[0](x) + self.resblocks[1](x) + self.resblocks[2](x)
        x = xs0 / 3

        # ---- Upsample 1 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[1](x)

        si = self.source_downs[1](s_stft)
        si = self.source_resblocks[1](si)
        x = x + si

        xs1 = self.resblocks[3](x) + self.resblocks[4](x) + self.resblocks[5](x)
        x = xs1 / 3

        # ---- Upsample 2 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[2](x)
        x = self.reflection_pad(x)

        si = self.source_downs[2](s_stft)
        si = self.source_resblocks[2](si)
        x = x + si

        xs2 = self.resblocks[6](x) + self.resblocks[7](x) + self.resblocks[8](x)
        x = xs2 / 3

        # ---- Final layers ----
        x = F.leaky_relu(x)
        x = self.conv_post(x)

        magnitude = torch.exp(x[:, :self.n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.n_fft // 2 + 1:, :])

        return magnitude, phase

    def forward(self, speech_tokens, speaker_embeddings, speaker_features):
        token_len = torch.full((speech_tokens.size(0),), speech_tokens.size(1), dtype=torch.long, device=speech_tokens.device)
        mask = (~make_pad_mask(token_len)).unsqueeze(-1)
        mel_len1, mel_len2, mu, spks, cond = self.flow_forward(speech_tokens, token_len, mask, speaker_embeddings, speaker_features)
        mu = mu.transpose(1, 2).contiguous()
        total_len = mel_len1.add(mel_len2).unsqueeze(0)
        mask = (~make_pad_mask(total_len)).unsqueeze(0)
        n_timesteps = 10
        temperature = 1.0
        x = torch.randn_like(mu, dtype=mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps+1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        dt_all = t_span[1:] - t_span[:-1]

        t = t_span[0:1]
        dt = dt_all[0:1]

        x_in = torch.cat([x, torch.zeros_like(x)], dim=0) 
        mask_in = torch.cat([mask, torch.zeros_like(mask)], dim=0) 
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0) 
        t_in = torch.cat([t, torch.zeros_like(t)], dim=0) 
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0) 
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        ## Classifier-Free Guidance inference introduced in VoiceBox
        # step 1
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[1 + 1] - t

        # step 2
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[2 + 1] - t

        # step 3
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[3 + 1] - t

        # step 4
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[4 + 1] - t

        # step 5
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[5 + 1] - t

        # step 6
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[6 + 1] - t

        # step 7
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[7 + 1] - t

        # step 8
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[8 + 1] - t

        # step 9
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[9 + 1] - t

        # step 10
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        output = x.float()
        speech_feat = torch.narrow(output, dim=2, start=mel_len1, length=output.size(2) - mel_len1)
        #mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        output_sources = s.transpose(1, 2).squeeze(1)
        spec = torch.stft(
            output_sources,
            self.n_fft, 
            self.hop_len, 
            self.n_fft, 
            window=self.stft_window.to(output_sources.device),
            return_complex=False)
        s_stft_real, s_stft_imag = spec[..., 0], spec[..., 1]
        output_sources = torch.cat([s_stft_real, s_stft_imag], dim=1)
        magnitude, phase = self.decode(x=speech_feat, s_stft=output_sources)
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        recombine_magnitude_phase = torch.cat([real, img], dim=1)
        output_wavs = self.istft(recombine_magnitude_phase)
        trim_fade = torch.zeros(2 * self.n_trim)
        cosine_window = (torch.cos(torch.linspace(torch.pi, 0, self.n_trim)) + 1) / 2
        trim_fade[self.n_trim:] = cosine_window
        output_wavs[:, :trim_fade.size(0)] *= trim_fade
        return output_wavs


@torch.no_grad()
def export_model_to_onnx(
    export_prepare_conditions=False, 
    export_cond_decoder=False, 
    audio_prompt_path=None, 
    output_export_dir=None, 
    output_file_name="output.wav", 
    device="cpu"):
    from chatterbox.tts import ChatterboxTTS
    if output_export_dir:
        import os
        os.makedirs(output_export_dir, exist_ok=True)

    chatterbox_model = ChatterboxTTS.from_pretrained(device=device)

    # replace DenseLayer of speake_encoder on custom SafeDenseLayer with exchanging BatchNorm1d layer on LayerNorm for ONNX export compatibility
    # we can safely do that because it does not affect inference as we do no need matching training dynamics
    # TODO Probably move this logic somewhere else outside export script
    old_dense = chatterbox_model.s3gen.speaker_encoder.xvector.dense
    chatterbox_model.s3gen.speaker_encoder.xvector.dense = SafeDenseLayer(old_dense.linear.in_channels, old_dense.linear.out_channels)
    chatterbox_model.s3gen.speaker_encoder.xvector.dense.linear.weight.copy_(old_dense.linear.weight)

    prepare_conditionals = PrepareConditionalsModel(chatterbox_model).eval()
    embed_tokens = InputsEmbeds(chatterbox_model).eval()
    cond_decoder = ConditionalDecoder(chatterbox_model).eval()

    audio_values = None
    if audio_prompt_path:
        audio_values, _sr = librosa.load(audio_prompt_path, sr=S3GEN_SR)
        audio_values = torch.from_numpy(audio_values).unsqueeze(0)

    input_ids=torch.tensor([[EXAGGERATION_TOKEN, 255, 281,  39,  46,  56,   2,  53,   2, 286,  41,  37,   2, 136, 122,
          49,   2, 152,   2, 103,   2, 277,  21, 101,   7,   2, 301,  55,  34,
          28,   7,   2,  53,   2, 296,  18,  18, 115,   2,  51,   2,  33, 245,
           2,  17, 190,   2,  42,   2,  50,  18, 125,   4,  32,   2, 290, 169,
         142,   2,  41,   2,  43,   2,  18,  29,  91,   2,  25, 186,   8,  20,
          14,  80,   2,  29,  86, 213, 216,   9,   0, START_SPEECH_TOKEN, START_SPEECH_TOKEN]])

    # NOTE: For some reason, the original implementation appends two speech tokens at the end
    # This is most likely by accident, but we keep it for compatibility
    position_ids = torch.where(
        input_ids >= START_SPEECH_TOKEN,
        0,
        torch.arange(input_ids.shape[1]).unsqueeze(0) - 1
    )

    exaggeration = torch.tensor([0.5])

    if export_prepare_conditions:
        torch.onnx.export(
            embed_tokens,
            (input_ids, position_ids, exaggeration),
            f"{output_export_dir}/embed_tokens.onnx",
            export_params=True,
            opset_version=20,
            input_names=["input_ids", "position_ids", "exaggeration"],
            output_names=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "position_ids": {0: "batch_size", 1: "sequence_length"},
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
                "exaggeration": {0: "batch_size"},
            },
        )
        print(f"✅ Embedding Tokens ONNX export is completed. Model saved as 'embed_tokens.onnx'")

        dummy_audio_values = torch.randn(1, 312936)
        torch.onnx.export(
            prepare_conditionals,
            (dummy_audio_values, ),
            f"{output_export_dir}/speech_encoder.onnx",
            export_params=True,
            opset_version=20,
            input_names=["audio_values"],
            output_names=["audio_features", "audio_tokens", "speaker_embeddings", "speaker_features"],
            dynamic_axes={
                "audio_values": {0: "batch_size", 1: "num_samples"},
                "audio_features": {0: "batch_size", 1: "sequence_length"},
                "audio_tokens": {0: "batch_size", 1: "audio_sequence_length"},
                "speaker_embeddings": {0: "batch_size"},
                "speaker_features": {
                    0: "batch_size",
                    1: "feature_dim",
                },
            },
        )
        print(f"✅ Speech Encoder ONNX export is completed. Model saved as 'speech_encoder.onnx'")


    # Example run
    # audio_values = torch.randn(1, 0) if not audio_values else audio_values
    cond_emb, prompt_token, speaker_embeddings, speaker_features = prepare_conditionals(audio_values=audio_values)

    text_emb = embed_tokens(input_ids=input_ids, position_ids=position_ids, exaggeration=exaggeration)

    inputs_embeds = torch.cat((cond_emb, text_emb), dim=1) # (B, length, dim)

    from transformers import LlamaForCausalLM, RepetitionPenaltyLogitsProcessor
    llm = LlamaForCausalLM.from_pretrained("vladislavbro/llama_backbone_0.5")
    llm.eval()

    repetition_penalty = 1.2
    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

    generate_tokens = torch.tensor([[START_SPEECH_TOKEN]], dtype=torch.long)
    max_new_tokens = 256
    past_key_values = None
    for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
        single_pass = llm(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )
        past_key_values = single_pass.past_key_values
        next_token_logits = single_pass.logits[:, -1, :]

        next_token_logits = repetition_penalty_processor(generate_tokens, next_token_logits)

        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generate_tokens = torch.cat((generate_tokens, next_token), dim=-1)
        if (next_token.view(-1) == STOP_SPEECH_TOKEN).all():
            break

        # embed next token
        position_ids = torch.full(
            (input_ids.shape[0], 1),
            i + 1,
            dtype=torch.long,
        )
        next_token_emb = embed_tokens(next_token, position_ids, exaggeration)
        inputs_embeds = next_token_emb

    speech_tokens = torch.cat([prompt_token, generate_tokens[:, 1:-1]], dim=1)

    if export_cond_decoder:
        torch.onnx.export(
            cond_decoder,
            (speech_tokens, speaker_embeddings, speaker_features),
            f"{output_export_dir}/conditional_decoder.onnx",
            export_params=True,
            opset_version=17,
            input_names=["speech_tokens", "speaker_embeddings", "speaker_features"],
            output_names=["waveform"],
            dynamic_axes={
                "speech_tokens": {
                    0: "batch_size",
                    1: "num_speech_tokens",
                },
                "speaker_embeddings": {
                    0: "batch_size",
                },
                "speaker_features": {
                    0: "batch_size",
                    1: "feature_dim",
                },
                "waveform": {0: 'batch_size', 1: 'num_samples'},
            }
        )
        print(f"✅ Conditional decoder ONNX export is completed. Model saved as 'conditional_decoder.onnx'")

    if export_prepare_conditions or export_cond_decoder:
        # https://github.com/inisis/OnnxSlim/issues/190#issuecomment-3314433214
        # for this optimization logic onnxslim==0.1.68 must be used
        os.environ['ONNXSLIM_THRESHOLD'] = '10000000000'
        import onnxslim
        import onnx
        for f in os.listdir(output_export_dir):
            if not f.endswith(".onnx"):
                continue
            save_path = os.path.join(output_export_dir, f)
            model = onnxslim.slim(save_path)
            onnx.save_model(model, save_path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(save_path) + "_data")

    output = cond_decoder(
        speech_tokens=speech_tokens,
        speaker_embeddings=speaker_embeddings,
        speaker_features=speaker_features,
    )

    ta.save(output_file_name, output, S3GEN_SR)
    print(f"{output_file_name} was successfully saved")


if __name__ == "__main__":
    AUDIO_PROMPT_PATH="path/to/audio.wav"
    export_model_to_onnx(
        export_prepare_conditions=False,
        export_cond_decoder=False,
        audio_prompt_path=AUDIO_PROMPT_PATH,
        output_export_dir="output_dir",
        output_file_name="output.wav"
    )
