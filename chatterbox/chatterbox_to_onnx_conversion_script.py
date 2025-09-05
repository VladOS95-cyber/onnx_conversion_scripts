# # !pip install --upgrade onnx==1.17.0 onnxruntime==1.20.1 onnxslim==0.1.48 transformers==4.48.3 chatterbox-tts==0.1.2

import torch
from torch import nn
import torch.nn.functional as F
from chatterbox.tts import ChatterboxTTS, T3Cond
import onnxslim
import os
from einops import pack, rearrange, repeat
import torchaudio as ta
from librosa.filters import mel as librosa_mel_fn
import torchaudio.compliance.kaldi as Kaldi

SPEECH_VOCAB_SIZE = 6561
SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1
CFM_PARAMS = {
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
}
S3GEN_SR = 24000
S3_SR = 16_000

# override certain torch functions
torch.Tensor.item = lambda x: x # no-op

def create_dummy_inputs(hp, batch_size=2, text_len=40):
    speaker_emb = torch.randn(batch_size, hp.speaker_embed_size)
    cond_prompt_speech_tokens = torch.randint(
        0, hp.speech_tokens_dict_size, (batch_size, hp.speech_cond_prompt_len)
    )
    emotion_adv = 0.5 * torch.ones(batch_size, 1, 1)
    text_tokens_ids = torch.randint(
        0, hp.text_tokens_dict_size, (batch_size, text_len)
    )
    sot = hp.start_text_token
    eot = hp.stop_text_token
    text_tokens_ids = F.pad(text_tokens_ids, (1, 0), value=sot)
    text_tokens_ids = F.pad(text_tokens_ids, (0, 1), value=eot)
    speech_input_ids = hp.start_speech_token * torch.ones_like(text_tokens_ids[:, :1])
    return speaker_emb, cond_prompt_speech_tokens, emotion_adv, text_tokens_ids, speech_input_ids

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
        max_len = max_len if max_len > 0 else lengths.max().item()
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

def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

def extract_feature(audio):
    features = []
    feature_times = []
    feature_lengths = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])
    # padding for batch inference
    features_padded = pad_list(features, pad_value=0)
    return features_padded, feature_lengths, feature_times


model = ChatterboxTTS.from_pretrained(device="cpu")
config = model.t3.hp

output_dir = "converted"
os.makedirs(output_dir, exist_ok=True)

# 1. Export speech encoder
class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_enc = model.t3.cond_enc
        self.text_emb = model.t3.text_emb
        self.speech_emb = model.t3.speech_emb
        self.speech_pos_emb = model.t3.speech_pos_emb
        self.text_pos_emb = model.t3.text_pos_emb
        self.hp = model.t3.hp
        self.perceiver = model.t3.cond_enc.perceiver
        self.spkr_enc = model.t3.cond_enc.spkr_enc
        self.emotion_adv_fc = model.t3.cond_enc.emotion_adv_fc
        self.register_buffer("bos_token", torch.tensor([[self.hp.start_speech_token]], dtype=torch.long))
    
    def conditional_encoding(self, cond: T3Cond):
        # Validate
        assert (cond.cond_prompt_speech_tokens is None) == (cond.cond_prompt_speech_emb is None), \
            "no embeddings for cond_prompt_speech_tokens"

        # Speaker embedding projection
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.hp.speaker_embed_size))[:, None]  # (B, 1, dim)
        empty = torch.zeros_like(cond_spkr[:, :0])  # (B, 0, dim)

        # TODO CLAP
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty  # (B, 0, dim)

        # Cond prompt
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion Adv: must provide a value if this model uses emotion conditioning
        cond_emotion_adv = empty  # (B, 0, dim)
        B = cond.emotion_adv.shape[0]
        if self.hp.emotion_adv:
            assert cond.emotion_adv is not None
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.reshape(B, 1, 1))

        # Concat and return
        cond_embeds = torch.cat((
            cond_spkr,
            cond_clap,
            cond_prompt_speech_emb,
            cond_emotion_adv,
        ), dim=1)
        return cond_embeds

    def forward(self, 
                speaker_emb, 
                cond_prompt_speech_tokens, 
                emotion_adv,
                text_tokens_ids,
                speech_input_ids):
        t3_cond = T3Cond(
            speaker_emb=speaker_emb,
            clap_emb=None,
            cond_prompt_speech_tokens=cond_prompt_speech_tokens,
            cond_prompt_speech_emb=None,
            emotion_adv=emotion_adv
            )
        text_tokens_ids = torch.atleast_2d(text_tokens_ids)
        t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
            self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        cond_emb = self.conditional_encoding(t3_cond)
        text_emb = self.text_emb(text_tokens_ids)
        text_emb[1::2].zero_() # CFG uncond
        # text_emb[1].zero_() # CFG uncond
        speech_emb = self.speech_emb(speech_input_ids)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens_ids)
            speech_emb = speech_emb + self.speech_pos_emb(speech_input_ids)
        len_cond = cond_emb.size(1)
        cond_emb = cond_emb.expand(text_emb.shape[0], -1, -1)

        # concat
        inputs_embeds = torch.cat((cond_emb, text_emb, speech_emb), dim=1) # (B, length, dim)
        bos_embed = self.speech_emb(self.bos_token)
        idx = torch.zeros((1, 1), dtype=torch.long, device=inputs_embeds.device)
        pos_embed =  self.speech_pos_emb(idx)
        bos_embed = bos_embed + pos_embed
        batch_size = inputs_embeds.size(0)
        bos_embed = bos_embed.expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([inputs_embeds, bos_embed], dim=1)
        return inputs_embeds, len_cond

speaker_emb, cond_prompt_speech_tokens, emotion_adv, text_tokens_ids, speech_input_ids = create_dummy_inputs(config)
embedder_model = SpeechEncoder()
inputs_embeds, _ = embedder_model(speaker_emb, cond_prompt_speech_tokens, emotion_adv, text_tokens_ids, speech_input_ids)
ort_speech_encoder_inputs = {
    "speaker_emb": speaker_emb.cpu().numpy(),
    "cond_prompt_speech_tokens": cond_prompt_speech_tokens.cpu().numpy(),
    "emotion_adv": emotion_adv.cpu().numpy(),
    "text_tokens_ids": text_tokens_ids.cpu().numpy(),
    "speech_input_ids": speech_input_ids.cpu().numpy()
}
for k, v in ort_speech_encoder_inputs.items():
    print(f'{k=}, {v.shape=}')
torch.onnx.export(
    embedder_model,
    (speaker_emb, cond_prompt_speech_tokens, emotion_adv, text_tokens_ids, speech_input_ids),
    f"{output_dir}/speech_encoder.onnx",
    export_params=True,
    opset_version=14,
    input_names=["speaker_emb", "cond_prompt_speech_tokens", "emotion_adv", "text_tokens_ids", "speech_input_ids"],
    output_names=["inputs_embeds", "len_cond"],
    dynamic_axes={
        "speaker_emb": {0: "batch_size"},
        "cond_prompt_speech_tokens": {0: "batch_size", 1: "seq_len_cond"},
        "emotion_adv": {0: "batch_size"},
        "text_tokens_ids": {0: "batch_size", 1: "seq_len_text"},
        "speech_input_ids": {0: "batch_size", 1: "seq_len_speech"},
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"}
    },
    # dynamo=True,
)
print(f"✅ Speech Encoder ONNX export is completed. Model saved as 'speech_encoder.onnx'")

ref_wav = torch.empty((1, 1000)).uniform_(-0.97619647, 0.93708616)
resampler = ta.transforms.Resample(S3GEN_SR, S3_SR)
# Tokenize 16khz reference
tokenizer = model.s3gen.tokenizer
# Resample to 16kHz
ref_wav_16 = resampler(ref_wav)
ref_speech_tokens, ref_speech_token_lens = tokenizer(ref_wav_16)
ref_wav_16, _, _ = extract_feature(ref_wav_16)

# 2. Export embedding reference wrapper
class EmbeddingReferenceWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.speaker_encoder = model.s3gen.speaker_encoder
    
    def mel_spectrogram(self, y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
        """Copied from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/audio.py
        Set default values according to Cosyvoice's config.
        """

        if len(y.shape) == 1:
            y = y[None, ]

        y = torch.nn.functional.pad(
            y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
        )
        y = y.squeeze(1)
        hann_window = torch.hann_window(1920)
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
        real = spec[..., 0]
        imag = spec[..., 1]
        spec = torch.sqrt(real**2 + imag**2 + 1e-9)
        spec = torch.matmul(mel, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1)

        return spec

    def forward(self, ref_wav, ref_wav_16, ref_speech_tokens, ref_speech_token_lens):
        ref_wav_24 = ref_wav
        ref_mels_24 = self.mel_spectrogram(ref_wav_24).transpose(1, 2)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder(ref_wav_16)

        # Make sure mel_len = 2 * stoken_len (happens when the input is not padded to multiple of 40ms)
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        prompt_token=ref_speech_tokens,
        prompt_token_len=ref_speech_token_lens,
        prompt_feat=ref_mels_24,
        embedding=ref_x_vector,
        return prompt_token, prompt_token_len, prompt_feat, embedding

embeddig_ref_wrapper = EmbeddingReferenceWrapper()
torch.onnx.export(
    embeddig_ref_wrapper,
    (ref_wav, ref_wav_16, ref_speech_tokens, ref_speech_token_lens),
    f"{output_dir}/embedding_ref.onnx",
    export_params=True,
    opset_version=17,
    input_names=["ref_wav", "ref_wav_16", "ref_speech_tokens", "ref_speech_token_lens"],
    output_names=["prompt_token", "prompt_token_len", "prompt_feat", "embedding"],
    dynamic_axes={
        "ref_wav": {1: "sequence_length"},
        "ref_wav_16": {1: "sequence_length"},
        "embedding": {1: "embedding_length"},
        "ref_speech_tokens": {1: "sequence_length"},
        "prompt_feat": {1: "feat_length"},
    },
)
print(f"✅ Embedding reference ONNX export is completed. Model saved as 'embedding_ref.onnx'")

# input('wait...')
batch_size, seq_len, _ = inputs_embeds.shape
num_layers = 30
num_key_value_heads = 16
head_dim = 64
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
dummy_past_key_values_kwargs = {
    f"past_key_values_{i}_{key}": torch.empty(
        batch_size, num_key_value_heads, seq_len, head_dim, dtype=torch.float32,
    )
    for i in range(num_layers)
    for key in ["key", "value"]
}
dynamic_axes = {
    "inputs_embeds": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "position_ids": {0: "batch_size", 1: "seq_len"}
}
pkv_input_names = list(dummy_past_key_values_kwargs.keys())
pkv_output_names = list(
    x.replace("past_key_values", "present") for x in dummy_past_key_values_kwargs.keys()
)
for name in pkv_input_names:
    dynamic_axes[name] = {0: "batch_size", 2: "past_seq_len"}
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
onnx_inputs = (inputs_embeds, *dummy_past_key_values_kwargs.values())
next_token = torch.full((batch_size, 1), 6563, dtype=torch.long)
idx = torch.tensor([0])

# 3. Export speech embedding
class SpeechEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.speech_emb = model.t3.speech_emb
        self.speech_pos_emb = model.t3.speech_pos_emb
    
    def forward(self, next_token, idx):
        next_token_embed = model.t3.speech_emb(next_token)
        next_token_embed = next_token_embed + model.t3.speech_pos_emb.get_fixed_embedding(idx + 1)
        return next_token_embed
    
speech_embedding = SpeechEmbedding()
torch.onnx.export(
    speech_embedding,
    (next_token, idx),
    f"{output_dir}/speech_embedding.onnx",
    export_params=True,
    opset_version=14,
    input_names=["next_token", "idx"],
    output_names=["next_token_embed"],
    dynamic_axes={
        "next_token": {0: "batch_size"},
    },
)
print(f"✅ Speech embedding ONNX export is completed. Model saved as 'speech_embedding.onnx'")

# 4. Export LLM Backbone (Llama3 from repo vladislavbro/llama_backbone_0.5) using https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py
# Before export, we replace embed_tokens nd lm_head of the LLM backbone model
# because it does not specifically use LLM tokens

# new_model = LlamaForCausalLM(config)
# new_model.model.load_state_dict(model.t3.tfmr.state_dict(), strict=True)
# new_model.config.vocab_size = model.t3.speech_head.out_features
# new_model.vocab_size = new_model.config.vocab_size
# new_model.model.embed_tokens = nn.Embedding(new_model.vocab_size, new_model.config.hidden_size)
# new_model.model.embed_tokens.weight.data.copy_(model.t3.speech_head.weight.data)
# new_model.lm_head = model.t3.speech_head
# new_model.tie_weights()

# new_model.push_to_hub("vladislavbro/llama_backbone_0.5", revision="refs/pr/1")

hidden_states = torch.randn(1, text_tokens_ids.shape[1], config.n_channels)
head_out = model.t3.speech_head(hidden_states)

speech_tokens = torch.randint(
        0, config.text_tokens_dict_size, (1, head_out.shape[1])
    )
speech_token_lens = torch.LongTensor([speech_tokens.size(1)])
speech_tokens, token_len = torch.concat([model.conds.gen["prompt_token"], speech_tokens], dim=1), model.conds.gen["prompt_token_len"] + speech_token_lens
mask = ~make_pad_mask(token_len).unsqueeze(-1).to(speech_tokens)
embedding = model.conds.gen["embedding"]
prompt_feat = model.conds.gen["prompt_feat"]

# 5. Export Flow inference wrapper
class FlowInferenceWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = model.s3gen.flow.output_size
        self.input_embedding = model.s3gen.flow.input_embedding
        self.spk_embed_affine_layer = model.s3gen.flow.spk_embed_affine_layer
        self.encoder = model.s3gen.flow.encoder
        self.encoder_proj = model.s3gen.flow.encoder_proj

    def forward(self, speech_tokens, token_len, mask, embedding, prompt_feat):

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
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size] ).to(text_encoded.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mu = text_encoded
        spks = embedding
        cond = conds
        return mel_len1, mel_len2, mu, spks, cond

flow_inference = FlowInferenceWrapper()
torch.onnx.export(
    flow_inference,
    (speech_tokens, token_len, mask, embedding, embedding),
    f"{output_dir}/flow_inference.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["speech_tokens", "token_len", "mask", "embedding", "prompt_feat"],
    output_names=["mel_len1", "mel_len2", "mu", "spks", "cond"],
    dynamic_axes={
        "speech_tokens": {
            0: "batch_size",
            1: "feature_dim"
        },

        "token_len": {
            0: "sequence_len"
        },

        "mask": {
            0: "batch_size", 
            1: "feature_dim",
            2: "time_steps" 
        },

        "embedding": {
            0: "batch_size", 
            1: "feature_dim",
        },

        "prompt_feat": {
            0: "batch_size", 
            1: "feature_dim",
        },
        
        "mu": {
            0: "batch_size", 
            1: "time_steps",
            2: "condition_dim" 
        },
        
        "spks": {
            0: "batch_size",
            1: "condition_dim",
            # Note: speaker embeddings typically have fixed feature dimension
        },
        
        "cond": {
            0: "batch_size", 
            1: "condition_dim",
            2: "time_steps"
        },
    }
)
print(f"✅ Flow inference ONNX export is completed. Model saved as 'flow_inference.onnx'")

#6. Export Conditional Decoder
class ConditionalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embeddings = model.s3gen.flow.decoder.estimator.time_embeddings
        self.time_mlp = model.s3gen.flow.decoder.estimator.time_mlp
        self.up_blocks = model.s3gen.flow.decoder.estimator.up_blocks
        self.static_chunk_size = model.s3gen.flow.decoder.estimator.static_chunk_size
        self.mid_blocks = model.s3gen.flow.decoder.estimator.mid_blocks
        self.down_blocks = model.s3gen.flow.decoder.estimator.down_blocks
        self.final_block = model.s3gen.flow.decoder.estimator.final_block
        self.final_proj = model.s3gen.flow.decoder.estimator.final_proj

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_to_bias(mask_down.bool() == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_to_bias(mask_mid.bool() == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_to_bias(mask_up.bool() == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask

cond_decoder = ConditionalDecoder()
mel_len1, mel_len2, mu, spks, cond = flow_inference(speech_tokens, token_len, mask)
mu = mu.transpose(1, 2).contiguous()
total_len = torch.tensor([mel_len1 + mel_len2])
mask = (~make_pad_mask(total_len)).squeeze(0)
rand_noise = torch.randn([1, 80, 50 * 300])
_, _, T = mu.shape
n_timesteps = 10
temperature = 1.0
x = rand_noise[:, :, :T].to(mu.device).to(mu.dtype) * temperature
t_span = torch.linspace(0, 1, n_timesteps+1, device=mu.device, dtype=mu.dtype)
t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
dt_all = t_span[1:] - t_span[:-1]
t = t_span[0:1]
x_in = torch.zeros([2, 80, T])
mask_in = torch.zeros([2, 1, T])
mu_in = torch.zeros([2, 80, T])
t_in = torch.zeros([2])
spks_in = torch.zeros([2, 80])
cond_in = torch.zeros([2, 80, T])
dt = dt_all[0:1]  # keep shape
x_in[:] = x
mask_in[:] = mask
mu_in[0] = mu
t_in[:] = t
spks_in[0] = spks
cond_in[0] = cond
torch.onnx.export(
    cond_decoder,
    (x_in, mask_in, mu_in, t_in, spks_in, cond_in),
    f"{output_dir}/conditional_decoder.onnx",
    export_params=True,
    opset_version=14,
    input_names=["x_in", "mask_in", "mu_in", "t_in", "spks_in", "cond_in"],
    output_names=["dphi_dt"],
    dynamic_axes={
        'x_in': {0: 'batch_size', 2: 'input_time'},
        'mask_in': {0: 'batch_size', 2: 'sequence_length'},
        'mu_in': {0: 'batch_size', 2: 'sequence_length'},
        't_in': {0: 'batch_size'},
        'spks_in': {0: 'batch_size'},
        "cond_in": {0: 'batch_size', 2: 'sequence_length'},
        "dphi_dt": {0: "batch_size", 2: 'sequence_length'}
    }
)
print(f"✅ Conditional decoder ONNX export is completed. Model saved as 'conditional_decoder.onnx'")

speech_feat = cond_decoder(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

#7. Export STFTWrapper
class STFTWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.istft_params = model.s3gen.mel2wav.istft_params
        self.stft_window = model.s3gen.mel2wav.stft_window
        self.f0_predictor = model.s3gen.mel2wav.f0_predictor
        self.f0_upsamp = model.s3gen.mel2wav.f0_upsamp
        self.m_source = model.s3gen.mel2wav.m_source
    
    def forward(self, speech_feat):
        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        output_sources = s.transpose(1, 2).squeeze(1)
        spec = torch.stft(
            output_sources,
            self.istft_params["n_fft"], 
            self.istft_params["hop_len"], 
            self.istft_params["n_fft"], 
            window=self.stft_window.to(output_sources.device),
            return_complex=False)
        s_stft_real, s_stft_imag = spec[..., 0], spec[..., 1]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        return s_stft

stft_wrapper = STFTWrapper()
output_sources = stft_wrapper(speech_feat)
torch.onnx.export(
    stft_wrapper,
    (speech_feat),
    f"{output_dir}/stft_wrapper.onnx",
    export_params=True,
    opset_version=17,
    input_names=["speech_feat"],
    output_names=["s_stft"],
    dynamic_axes={
        'speech_feat': {0: 'batch_size', 2: 'sequence_length'},
        's_stft': {0: 'batch_size', 1: 'frequency_samples', 2: 'number_of_frames'},
    }
)
print(f"✅ STFTWrapper ONNX export is completed. Model saved as 'stft_wrapper.onnx'")

#8. Export HiFTGenerator
class HiFTGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.istft_params = model.s3gen.mel2wav.istft_params
        self.conv_pre = model.s3gen.mel2wav.conv_pre
        self.num_upsamples = model.s3gen.mel2wav.num_upsamples
        self.lrelu_slope = model.s3gen.mel2wav.lrelu_slope
        self.reflection_pad = model.s3gen.mel2wav.reflection_pad
        self.ups = model.s3gen.mel2wav.ups
        self.source_downs = model.s3gen.mel2wav.source_downs
        self.source_resblocks = model.s3gen.mel2wav.source_resblocks
        self.num_kernels = model.s3gen.mel2wav.num_kernels
        self.resblocks = model.s3gen.mel2wav.resblocks
        self.conv_post = model.s3gen.mel2wav.conv_post

    def decode(self, x: torch.Tensor, s_stft: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # actually, sin is redundancy
        return magnitude, phase
    
    def forward(self, speech_feat, output_sources):
        magnitude, phase = self.decode(x=speech_feat, s_stft=output_sources)
        return magnitude, phase

hift_generator = HiFTGenerator()
torch.onnx.export(
    hift_generator,
    (speech_feat, output_sources),
    f"{output_dir}/hift_generator.onnx",
    export_params=True,
    opset_version=14,
    input_names=["speech_feat", "output_sources"],
    output_names=["magnitude", "phase"],
    dynamic_axes={
        'speech_feat': {0: 'batch_size', 2: 'sequence_length'},
        "output_sources": {0: 'batch_size', 1: 'frequency_samples', 2: 'number_of_frames'},
        "magnitude": {0: 'batch_size', 1: 'frequency_samples', 2: 'number_of_frames'},
        "phase": {0: 'batch_size', 1: 'frequency_samples', 2: 'number_of_frames'},
    }
)
print(f"✅ HiFTGenerator ONNX export is completed. Model saved as 'hift_generator.onnx'")

#9. Post-processing
for f in os.listdir(output_dir):
    p = os.path.join(output_dir, f)
    onnxslim.slim(p, p)
print("Chatterbox model export successfully completed")
