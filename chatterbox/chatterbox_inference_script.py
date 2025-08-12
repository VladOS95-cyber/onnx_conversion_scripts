import onnxruntime
import numpy as np
from huggingface_hub import hf_hub_download
from chatterbox.tts import ChatterboxTTS, Conditionals
import torchaudio as ta
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper
from tqdm import tqdm
from scipy.signal import get_window
import perth

SPACE = "[SPACE]"
SPEECH_VOCAB_SIZE = 6561
SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1
S3GEN_SR = 24000
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

def update_llama_input_with_new_past(llama_input, past_key_values_output):
    """
    Updates the llama_input dictionary with the new past_key_values
    returned from the ONNX model forward pass.

    Args:
        llama_input (dict): The original input dict for the model.
        past_key_values_output (list): The 60 output tensors: 30 keys and 30 values.
    
    Returns:
        dict: Updated llama_input with new past_key_values.
    """
    assert len(past_key_values_output) == 60, "Expected 60 past_key/value tensors (30 layers)."

    updated_input = llama_input.copy()  # Don't mutate original

    for layer in range(30):
        key_name = f'past_key_values.{layer}.key'
        value_name = f'past_key_values.{layer}.value'
        
        key_tensor = past_key_values_output[2 * layer]
        value_tensor = past_key_values_output[2 * layer + 1]

        updated_input[key_name] = key_tensor
        updated_input[value_name] = value_tensor

    return updated_input

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
        max_len = torch.maximum(lengths.max(), torch.tensor(max_len))
        seq_range = torch.arange(0,
                                    max_len,
                                    dtype=torch.int64,
                                    device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = lengths.unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
        return mask

# 1. Load model
output_dir = "converted"
model_id = "vladislavbro/chatterbox_ONNX"
# llama_backbone_path = hf_hub_download(repo_id=model_id, filename="llama3.onnx", local_dir=output_dir)
# hf_hub_download(repo_id=model_id, filename="llama3.data", local_dir=output_dir)
# hf_hub_download(repo_id=model_id, filename="genai_config.json", local_dir=output_dir)
speech_encoder_path = hf_hub_download(repo_id=model_id, filename="speech_encoder.onnx", local_dir=output_dir)
conditional_docoder_path = hf_hub_download(repo_id=model_id, filename="conditional_decoder.onnx", local_dir=output_dir)
flow_inference_path = hf_hub_download(repo_id=model_id, filename="flow_inference.onnx", local_dir=output_dir)
stft_wrapper_path = hf_hub_download(repo_id=model_id, filename="stft_wrapper.onnx", local_dir=output_dir)
hift_generator_path = hf_hub_download(repo_id=model_id, filename="hift_generator.onnx", local_dir=output_dir)
hf_hub_download(repo_id="ResembleAI/chatterbox", filename="tokenizer.json", local_dir=output_dir)
hf_hub_download(repo_id="ResembleAI/chatterbox", filename="conds.pt", local_dir=output_dir)

#2. Start inferense sessions
llama_session = onnxruntime.InferenceSession("converted/model.onnx")
cond_decoder_session = onnxruntime.InferenceSession(conditional_docoder_path)
speech_encoder_session = onnxruntime.InferenceSession("converted/speech_encoder.onnx")
flow_inference_session = onnxruntime.InferenceSession(flow_inference_path)
stft_wrapper_session = onnxruntime.InferenceSession(stft_wrapper_path)
hift_generator_session = onnxruntime.InferenceSession(hift_generator_path)

#3. Prepare input
tokenizer = Tokenizer.from_file("converted/tokenizer.json")
model = ChatterboxTTS.from_pretrained(device="cpu")
text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
#wav = model.generate(text)
#ta.save("test-1.wav", wav, model.sr)
text = text.replace(' ', SPACE)
text_tokens_ids = tokenizer.encode(text).ids
text_tokens_ids = torch.IntTensor(text_tokens_ids).unsqueeze(0)
text_tokens_ids = torch.cat([text_tokens_ids, text_tokens_ids], dim=0)
text_tokens_ids = F.pad(text_tokens_ids, (1, 0), value=start_text_token)
text_tokens_ids = F.pad(text_tokens_ids, (0, 1), value=stop_text_token)
text_tokens_ids = torch.atleast_2d(text_tokens_ids).to(dtype=torch.long)
speech_input_ids = start_speech_token * torch.ones_like(text_tokens_ids[:, :1])
emotion_adv= 0.5 * torch.ones(1, 1, 1)
conds = Conditionals.load("converted/conds.pt")
cond_prompt_speech_tokens = conds.t3.cond_prompt_speech_tokens
speaker_emb = conds.t3.speaker_emb

ort_speech_encoder_inputs = {
    "speaker_emb": speaker_emb.cpu().numpy(),
    "cond_prompt_speech_tokens": cond_prompt_speech_tokens.cpu().numpy(),
    "emotion_adv": emotion_adv.cpu().numpy(),
    "text_tokens_ids": text_tokens_ids.cpu().numpy(),
    "speech_input_ids": speech_input_ids.cpu().numpy()
}
inputs_embeds, len_cond = speech_encoder_session.run(None, ort_speech_encoder_inputs)

#4. Instantiate the logits processors.
min_p=0.05
top_p=1.00
repetition_penalty=1.2
min_p_warper = MinPLogitsWarper(min_p=min_p)
top_p_warper = TopPLogitsWarper(top_p=top_p)
repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
num_layers = 30
num_key_value_heads = 16
head_dim = 64
batch_size, seq_len, _ = inputs_embeds.shape
dtype = np.float32
cfg_weight=0.5
temperature=0.8
max_new_tokens = 150
dummy_attention_mask = np.ones(inputs_embeds.shape[:2], dtype=np.int64)
llama_input = {
    "inputs_embeds": inputs_embeds,
    "attention_mask": dummy_attention_mask
}
for layer in range(num_layers):
    key_name = f'past_key_values.{layer}.key'
    value_name = f'past_key_values.{layer}.value'
    
    # Shape: (batch_size, num_key_value_heads, past_seq_len, head_dim)
    shape = (batch_size, num_key_value_heads, seq_len, head_dim)
    
    llama_input[key_name] = np.empty(shape, dtype=dtype)
    llama_input[value_name] = np.empty(shape, dtype=dtype)

#5. Token generation 
llama_out = llama_session.run(None, llama_input)
past_key_values = llama_out[1:]
predicted = []
generated_ids = torch.tensor([[start_speech_token]], dtype=torch.long)
# ---- Generation Loop using kv_cache ----
for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
    logits = llama_out[0]
    logits = logits[:, -1, :]
    logits = torch.from_numpy(logits)

    # CFG
    if cfg_weight > 0.0:
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

    predicted.append(next_token)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Check for EOS token.
    if next_token.view(-1) == stop_speech_token:
        break

    # Get embedding for the new token.
    next_token_embed = model.t3.speech_emb(next_token)
    next_token_embed = next_token_embed + model.t3.speech_pos_emb.get_fixed_embedding(i + 1)

    #  For CFG
    if cfg_weight > 0.0:
        next_token_embed = torch.cat([next_token_embed, next_token_embed])

    # Forward pass with only the new token and the cached past.
    llama_input = update_llama_input_with_new_past(llama_input, past_key_values)
    llama_input["inputs_embeds"] = next_token_embed.detach().numpy()
    llama_out = llama_session.run(None, llama_input)

    # Update the kv_cache.
    past_key_values = llama_out[1:]

# Concatenate all predicted tokens along the sequence dimension.
predicted_tokens = torch.cat(predicted, dim=1)  # shape: (B, num_tokens)
# Extract only the conditional batch.
speech_tokens = predicted_tokens[0]

speech_tokens = drop_invalid_tokens(speech_tokens)
speech_tokens = speech_tokens[speech_tokens < 6561]
speech_tokens = speech_tokens.unsqueeze(0)
speech_token_lens = torch.LongTensor([speech_tokens.size(1)])
speech_tokens, token_len = torch.concat([conds.gen["prompt_token"], speech_tokens], dim=1), conds.gen["prompt_token_len"] + speech_token_lens
mask = ~make_pad_mask(token_len).unsqueeze(-1).to(speech_tokens)
flow_infer_input = {
    "speech_tokens": speech_tokens.detach().numpy(),
    "token_len": token_len.detach().numpy(),
    "mask": mask.detach().numpy()
}
mel_len1, mel_len2, mu, spks, cond = flow_inference_session.run(None, flow_infer_input)

#6. Conditional decoding
total_len = torch.tensor([mel_len1 + mel_len2])
mask = (~make_pad_mask(total_len)).squeeze(0).detach().numpy()
rand_noise = torch.randn([1, 80, 50 * 300])
B, _, T = mu.shape
n_timesteps = 10
temperature = 1.0
x = rand_noise[:, :, :T].to(mu.device).detach().numpy() * temperature
t_span = torch.linspace(0, 1, n_timesteps+1, device=mu.device)
t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
dt_all = (t_span[1:] - t_span[:-1]).detach().numpy()
t = t_span[0:1].detach().numpy()
x_in = np.zeros([2, 80, T])
mask_in = np.zeros([2, 1, T])
mu_in = np.zeros([2, 80, T])
t_in = np.zeros([2])
spks_in = np.zeros([2, 80])
cond_in = np.zeros([2, 80, T])
sol = []
for i in range(n_timesteps):
    dt = dt_all[i:i + 1]  # keep shape
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
ta.save("output.wav", watermarked_wav, S3GEN_SR)
print("output.wav was successfully saved")