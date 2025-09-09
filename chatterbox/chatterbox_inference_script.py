import onnxruntime
import numpy as np
from huggingface_hub import hf_hub_download
from chatterbox.tts import Conditionals
import torchaudio as ta
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper
from tqdm import tqdm
from scipy.signal import get_window
import perth
import librosa

SPACE = "[SPACE]"
SPEECH_VOCAB_SIZE = 6561
SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1
S3GEN_SR = 24000
# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
DEC_COND_LEN = 10 * S3GEN_SR
start_text_token = 255
stop_text_token = 0
start_speech_token = 6561
stop_speech_token = 6562
CFM_PARAMS = {
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
}
n_timesteps = 10
temperature = 1.0

def drop_invalid_tokens(x):
    """Drop SoS and EoS"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
    if SOS in x:
        s = (x == SOS).nonzero(as_tuple=True)[0].squeeze(0) + 1
    else:
        s = 0

    if EOS in x:
        e = (x == EOS).nonzero(as_tuple=True)[0].squeeze(0)
    else:
        e = None

    x = x[s: e]
    return x

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
    max_len = max(x.shape[0] for x in xs)

    # shape = (B, Tmax, *) using the first tensor’s remaining dimensions
    out_shape = (n_batch, max_len) + xs[0].shape[1:]
    pad = np.full(out_shape, pad_value, dtype=xs[0].dtype)

    for i, x in enumerate(xs):
        length = x.shape[0]
        pad[i, :length, ...] = x

    return pad

output_dir = "converted"
output_file_name = "infer_output.wav"
model_id = "onnx-community/chatterbox-onnx"
text = "The Lord of the Rings is the greatest work of literature."
audio = None
target_voice_path = "back_4_more_fun.wav"

## Load model
speech_encoder_path = hf_hub_download(repo_id=model_id, filename="speech_encoder.onnx", local_dir=output_dir)
speech_embedding_path = hf_hub_download(repo_id=model_id, filename="speech_embedding.onnx", local_dir=output_dir)
language_model_path = hf_hub_download(repo_id=model_id, filename="language_model.onnx", local_dir=output_dir, subfolder='onnx')
hf_hub_download(repo_id=model_id, filename="language_model.onnx_data", local_dir=output_dir, subfolder='onnx')
conditional_decoder_path = hf_hub_download(repo_id=model_id, filename="conditional_decoder.onnx", local_dir=output_dir)
flow_inference_path = hf_hub_download(repo_id=model_id, filename="flow_inference.onnx", local_dir=output_dir)
stft_wrapper_path = hf_hub_download(repo_id=model_id, filename="stft_wrapper.onnx", local_dir=output_dir)
hift_generator_path = hf_hub_download(repo_id=model_id, filename="hift_generator.onnx", local_dir=output_dir)
tokenizer_path = hf_hub_download(repo_id=model_id, filename="speech_tokenizer_v2.onnx", local_dir=output_dir)
hf_hub_download(repo_id=model_id, filename="tokenizer.json", local_dir=output_dir)
hf_hub_download(repo_id=model_id, filename="conds.pt", local_dir=output_dir)
hf_hub_download(repo_id=model_id, filename="tokenizer.json", local_dir=output_dir)
hf_hub_download(repo_id=model_id, filename="vc_tokenizer_weights.pth", local_dir=output_dir)
conds = Conditionals.load("converted/conds.pt")

## Start common inferense sessions
tokenizer_session = onnxruntime.InferenceSession(tokenizer_path)
flow_inference_session = onnxruntime.InferenceSession(flow_inference_path)
stft_wrapper_session = onnxruntime.InferenceSession(stft_wrapper_path)
hift_generator_session = onnxruntime.InferenceSession(hift_generator_path)
cond_decoder_session = onnxruntime.InferenceSession(conditional_decoder_path)

def set_target_voice(wav_fpath):
    feature_extractor_path = hf_hub_download(repo_id=model_id, filename="feature_extractor.onnx", local_dir=output_dir)
    speaker_emb_path = hf_hub_download(repo_id=model_id, filename="speaker_emb.onnx", local_dir=output_dir)
    feature_extractor_session = onnxruntime.InferenceSession(feature_extractor_path)
    speaker_encoder_session = onnxruntime.InferenceSession(speaker_emb_path)
    s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
    s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
    if len(s3gen_ref_wav.shape) == 1: 
        s3gen_ref_wav = np.expand_dims(s3gen_ref_wav, axis=0)
    feature_ort_input = {
        "ref_wav": s3gen_ref_wav
    }
    feature, ref_mels_24, ref_wav_16 = feature_extractor_session.run(None, feature_ort_input)
    padded_feature = pad_list([feature], 0)
    speaker_enc_input = {
        "feature": padded_feature
    }
    embedding = speaker_encoder_session.run(None, speaker_enc_input)[0]
    feats_length = np.array([ref_wav_16.shape[-1]], dtype=np.int32)
    prompt_token_len = (feats_length + 2 - 1 * (3 - 1) - 1) // 2 + 1
    prompt_token_len = (prompt_token_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
    ort_tokenizer_input = {
        "feats": ref_wav_16,
        "feats_length": feats_length
    }
    ref_speech_tokens = tokenizer_session.run(None, ort_tokenizer_input)[0]
    ref_dict = {
        "prompt_token": ref_speech_tokens,
        "prompt_token_len": prompt_token_len,
        "prompt_feat": ref_mels_24,
        "embedding" : embedding
    }
    return ref_dict

def set_ref_dict(target_voice_path=None):
    ref_dict = {key: val.detach().numpy() for key, val in conds.gen.items() if val is not None }
    if target_voice_path:
        ref_dict = set_target_voice(target_voice_path)
    return ref_dict

ref_dict = set_ref_dict(target_voice_path)

# def execute_audio_to_audio_inference(audio, ref_dict):
#     print("Start Audio to Audio inference script...")
#     ## Prepare input
#     audio_16, _ = librosa.load(audio, sr=S3_SR)
#     audio_16 = torch.from_numpy(audio_16).float()[None, ]
#     speech_tokens, _ = tokenizer(audio_16)
#     speech_token_lens = np.array(speech_tokens.size(1))
#     prompt_token = ref_dict["prompt_token"]
#     speech_tokens, token_len = np.concatenate([prompt_token, speech_tokens], axis=1), ref_dict["prompt_token_len"] + speech_token_lens
#     return speech_tokens, token_len

def execute_text_to_audio_inference(text, ref_dict, conds):
    print("Start Text to Audio inference script...")
    ## Start inferense sessions
    llama_with_past_session = onnxruntime.InferenceSession(language_model_path)
    speech_encoder_session = onnxruntime.InferenceSession(speech_encoder_path)
    speech_embedding_session = onnxruntime.InferenceSession(speech_embedding_path)

    ## Prepare input
    tokenizer = Tokenizer.from_file(f"{output_dir}/tokenizer.json")
    text = text.replace(' ', SPACE)
    text_tokens_ids = tokenizer.encode(text).ids
    text_tokens_ids = torch.IntTensor(text_tokens_ids).unsqueeze(0)
    text_tokens_ids = torch.cat([text_tokens_ids, text_tokens_ids], dim=0)
    text_tokens_ids = F.pad(text_tokens_ids, (1, 0), value=start_text_token)
    text_tokens_ids = F.pad(text_tokens_ids, (0, 1), value=stop_text_token)
    text_tokens_ids = torch.atleast_2d(text_tokens_ids).to(dtype=torch.long)
    speech_input_ids = start_speech_token * torch.ones_like(text_tokens_ids[:, :1])
    emotion_adv= 0.5 * torch.ones(1, 1, 1)
    cond_prompt_speech_tokens = conds.t3.cond_prompt_speech_tokens
    speaker_emb = conds.t3.speaker_emb

    ort_speech_encoder_inputs = {
        "speaker_emb": speaker_emb.cpu().numpy(),
        "cond_prompt_speech_tokens": cond_prompt_speech_tokens.cpu().numpy(),
        "emotion_adv": emotion_adv.cpu().numpy(),
        "text_tokens_ids": text_tokens_ids.cpu().numpy(),
        "speech_input_ids": speech_input_ids.cpu().numpy()
    }
    inputs_embeds, _ = speech_encoder_session.run(None, ort_speech_encoder_inputs)

    ## Instantiate the logits processors.
    min_p=0.05
    top_p=1.00
    repetition_penalty=1.2
    min_p_warper = MinPLogitsWarper(min_p=min_p)
    top_p_warper = TopPLogitsWarper(top_p=top_p)
    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

    num_hidden_layers = 30
    num_key_value_heads = 16
    head_dim = 64
    batch_size, seq_len, _ = inputs_embeds.shape
    cfg_weight=0.5
    temperature=1e-8
    max_new_tokens = 128

    ## Prepare decoder inputs
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    position_ids = np.cumsum(attention_mask, axis=1, dtype=np.int64) - 1
    batch_size = inputs_embeds.shape[0]
    past_key_values = {
        f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
        for layer in range(num_hidden_layers)
        for kv in ("key", "value")
    }


    generated_tokens = []
    bos_token = torch.tensor([[start_speech_token]], dtype=torch.long)
    generated_ids = bos_token.clone()

    # ---- Generation Loop using kv_cache ----
    for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
        logits, *present_key_values = llama_with_past_session.run(None, dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **past_key_values,
        ))

        # TODO: UPDATE (use numpy where possible)
        logits = torch.from_numpy(logits)
        logits = logits[:, -1, :]
        if cfg_weight > 0.0: # CFG
            logits_cond = logits[0:1]
            logits_uncond = logits[1:2]
            logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

        logits = logits.squeeze(1)

        # Apply temperature scaling.
        if temperature != 1.0:
            logits = logits / temperature

        # Apply repetition penalty and top‑p filtering.
        logits = repetition_penalty_processor(generated_ids, logits)
        logits = min_p_warper(None, logits)
        logits = top_p_warper(None, logits)

        # Convert logits to probabilities and sample the next token.
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
        generated_tokens.append(next_token)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Check for EOS token.
        if (next_token.view(-1) == stop_speech_token).all():
            break

        # Get embedding for the new token.
        speech_embd_input = {
            "next_token": next_token.detach().numpy(),
            "idx" : np.array([i])
        }
        next_token_embed = speech_embedding_session.run(None, speech_embd_input)
        next_token_embed = torch.from_numpy(next_token_embed[0])

        #  For CFG
        if cfg_weight > 0.0:
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

        ## Update values for next generation loop
        attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
        position_ids = position_ids[:, -1:] + 1
        for j, key in enumerate(past_key_values):
            past_key_values[key] = present_key_values[j]

        inputs_embeds = next_token_embed.detach().numpy()

        # Concatenate all predicted tokens along the sequence dimension.
    predicted_tokens = torch.cat(generated_tokens, dim=1)  # shape: (B, num_tokens)

    # Extract only the conditional batch.
    speech_tokens = predicted_tokens[0]

    speech_tokens = drop_invalid_tokens(speech_tokens)
    speech_tokens = speech_tokens[speech_tokens < 6561]
    speech_tokens = speech_tokens.unsqueeze(0)
    speech_token_lens = np.array(speech_tokens.size(1))
    prompt_token = ref_dict["prompt_token"]
    speech_tokens, token_len = np.concatenate([prompt_token, speech_tokens], axis=1), ref_dict["prompt_token_len"] + speech_token_lens
    return speech_tokens, token_len

speech_tokens, token_len = None, None
# if audio:
#     speech_tokens, token_len = execute_audio_to_audio_inference(audio, ref_dict)
# else:
speech_tokens, token_len = execute_text_to_audio_inference(text, ref_dict, conds)

mask = (~make_pad_mask(torch.from_numpy(token_len))).unsqueeze(-1)
flow_infer_input = {
    "speech_tokens": speech_tokens,
    "token_len": token_len,
    "mask": mask.detach().numpy().astype(np.int64),
    "embedding": ref_dict["embedding"],
    "prompt_feat": ref_dict["prompt_feat"],
}
mel_len1, mel_len2, mu, spks, cond = flow_inference_session.run(None, flow_infer_input)
mu = np.ascontiguousarray(np.transpose(mu, (0, 2, 1)))

## Conditional decoding
total_len = torch.tensor([mel_len1 + mel_len2])
mask = (~make_pad_mask(total_len)).squeeze(0).detach().numpy()
rand_noise = torch.randn([1, 80, 50 * 300])
B, _, T = mu.shape
n_timesteps = 10
temperature = 1.0
x = rand_noise[:, :, :T].to(mu.device).detach().numpy() * temperature
t_span = torch.linspace(0, 1, n_timesteps+1, device=mu.device)
t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
t_span = t_span.detach().numpy()
t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
x_in = np.zeros([2, 80, T])
mask_in = np.zeros([2, 1, T])
mu_in = np.zeros([2, 80, T])
t_in = np.zeros([2])
spks_in = np.zeros([2, 80])
cond_in = np.zeros([2, 80, T])
sol = []
for i in range(1, len(t_span)):
    # Classifier-Free Guidance inference introduced in VoiceBox
    x_in[:] = x
    mask_in[:] = mask
    mu_in[0] = mu
    t_in[:] = t
    spks_in[0] = spks
    cond_in[0] = cond
    ort_cond_decoder_input = {
        "x_in": x_in.astype(np.float32), 
        "mask_in": mask_in.astype(np.float32), 
        "mu_in": mu_in.astype(np.float32), 
        "t_in": t_in.astype(np.float32), 
        "spks_in": spks_in.astype(np.float32),
        "cond_in": cond_in.astype(np.float32)
    }
    dphi_dt = cond_decoder_session.run(None, ort_cond_decoder_input)
    dphi_dt, cfg_dphi_dt = torch.split(torch.from_numpy(dphi_dt[0]), [B, B], dim=0)
    dphi_dt = (1.0 + CFM_PARAMS["inference_cfg_rate"]) * dphi_dt - CFM_PARAMS["inference_cfg_rate"] * cfg_dphi_dt

    x = x + dt * dphi_dt.detach().numpy()
    t = t + dt
    sol.append(x)
    if i < len(t_span) - 1:
        dt = t_span[i + 1] - t

output_mels = sol[-1]
output_mels = output_mels[:, :, mel_len1.item():]

stft_ort_wrapper_input = {
    "speech_feat": output_mels
}
output_sources = stft_wrapper_session.run(None, stft_ort_wrapper_input)
istft_params = {"n_fft": 16, "hop_len": 4}
def _istft(magnitude, phase):
    stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
    magnitude = torch.clip(magnitude, max=1e2)
    real = magnitude * torch.cos(phase)
    img = magnitude * torch.sin(phase)
    inverse_transform = torch.istft(torch.complex(real, img), istft_params["n_fft"], istft_params["hop_len"],
                                    istft_params["n_fft"], window=stft_window.to(magnitude.device))
    return inverse_transform

hift_generator_ort_input = {
    "speech_feat": output_mels,
    "output_sources": output_sources[0]
}
magnitude, phase = hift_generator_session.run(None, hift_generator_ort_input)
output_wavs = _istft(torch.from_numpy(magnitude), torch.from_numpy(phase))
n_trim = S3GEN_SR // 50  # 20ms = half of a frame
trim_fade = torch.zeros(2 * n_trim)
trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
output_wavs[:, :len(trim_fade)] *= trim_fade
wav = output_wavs.squeeze(0).detach().cpu().numpy()
watermarker = perth.PerthImplicitWatermarker()
watermarked_wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)
watermarked_wav = torch.from_numpy(watermarked_wav).unsqueeze(0)
ta.save(output_file_name, watermarked_wav, S3GEN_SR)
print(f"{output_file_name} was successfully saved")
