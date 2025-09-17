import onnxruntime
import numpy as np
from huggingface_hub import hf_hub_download
from chatterbox.tts import Conditionals
import torchaudio as ta
import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from tqdm import tqdm
import perth
import librosa
from transformers import AutoTokenizer
import torch.nn.functional as F

SPACE = "[SPACE]"
SPEECH_VOCAB_SIZE = 6561
SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1
S3GEN_SR = 24000
# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
DEC_COND_LEN = 10 * S3GEN_SR
start_speech_token = 6561
stop_speech_token = 6562

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

output_dir = "converted"
output_file_name = "output.wav"
model_id = "onnx-community/chatterbox-onnx"
text = "The Lord of the Rings is the greatest work of literature."
audio = None
target_voice_path = None

## Load model
speech_encoder_path = hf_hub_download(repo_id=model_id, filename="speech_encoder.onnx", local_dir=output_dir, subfolder='onnx')
language_model_path = hf_hub_download(repo_id=model_id, filename="language_model.onnx", local_dir=output_dir, subfolder='onnx')
embed_tokens_path = hf_hub_download(repo_id=model_id, filename="embed_tokens.onnx", local_dir=output_dir, subfolder='onnx')
hf_hub_download(repo_id=model_id, filename="language_model.onnx_data", local_dir=output_dir, subfolder='onnx')
conditional_decoder_path = hf_hub_download(repo_id=model_id, filename="conditional_decoder.onnx", local_dir=output_dir, subfolder='onnx')
hf_hub_download(repo_id=model_id, filename="conds.pt", local_dir=output_dir)
conds = Conditionals.load("converted/conds.pt")

## Start common inferense sessions
cond_decoder_session = onnxruntime.InferenceSession(conditional_decoder_path)
speech_encoder_session = onnxruntime.InferenceSession(speech_encoder_path)
embed_tokens_session = onnxruntime.InferenceSession(embed_tokens_path)

def set_ref_dict(wav_fpath):
    ort_speech_encoder_input = {
        "audio_values": np.empty((1, 0), dtype=np.float32),
        "exaggeration": np.array(0.5, dtype=np.float32)
    }
    if not wav_fpath:
        cond_emb, prompt_token, ref_x_vector, prompt_feat = speech_encoder_session.run(None, ort_speech_encoder_input)
        ref_dict = {
            "prompt_token": prompt_token,
            "cond_emb": cond_emb,
            "prompt_feat": prompt_feat,
            "embedding" : ref_x_vector
        }
        return ref_dict
    audio_values, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
    audio_values = torch.from_numpy(audio_values).unsqueeze(0)
    ort_speech_encoder_input["audio_values"] = audio_values.detach().cpu().numpy().astype(np.float32)
    cond_emb, prompt_token, ref_x_vector, prompt_feat = speech_encoder_session.run(None, ort_speech_encoder_input)
    ref_dict = {
        "prompt_token": prompt_token,
        "cond_emb": cond_emb,
        "prompt_feat": prompt_feat,
        "embedding" : ref_x_vector
    }
    return ref_dict

ref_dict = set_ref_dict(target_voice_path)

def execute_text_to_audio_inference(text, ref_dict):
    print("Start Text to Audio inference script...")
    ## Start inferense sessions
    llama_with_past_session = onnxruntime.InferenceSession(language_model_path)

    ## Prepare input
    text = text.replace(' ', SPACE)
    tokenizer = AutoTokenizer.from_pretrained("onnx-community/chatterbox-onnx")
    text_tokens_ids = tokenizer(text)["input_ids"]
    text_tokens_ids = torch.IntTensor(text_tokens_ids).unsqueeze(0)
    input_ids = F.pad(text_tokens_ids, (0, 1), value=start_speech_token)
    input_ids = F.pad(input_ids, (0, 1), value=start_speech_token)
    position_ids = torch.where(
        input_ids == start_speech_token,
        0,
        torch.arange(input_ids.shape[1]).unsqueeze(0)
    )

    ort_embed_tokens_inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int64),
        "position_ids": position_ids.cpu().numpy(),
    }
    inputs_embeds = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]
    inputs_embeds = np.concat((ref_dict["cond_emb"], inputs_embeds), axis=1)

    ## Instantiate the logits processors.
    repetition_penalty=1.2
    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

    num_hidden_layers = 30
    num_key_value_heads = 16
    head_dim = 64
    batch_size, seq_len, _ = inputs_embeds.shape
    max_new_tokens = 128

    ## Prepare llm inputs
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    llm_position_ids = np.cumsum(attention_mask, axis=1, dtype=np.int64) - 1
    batch_size = inputs_embeds.shape[0]
    past_key_values = {
        f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
        for layer in range(num_hidden_layers)
        for kv in ("key", "value")
    }

    generate_tokens = torch.tensor([[start_speech_token]], dtype=torch.long)

    # ---- Generation Loop using kv_cache ----
    for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
        logits, *present_key_values = llama_with_past_session.run(None, dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=llm_position_ids,
            **past_key_values,
        ))

        # TODO: UPDATE (use numpy where possible)
        logits = torch.from_numpy(logits)
        logits = logits[:, -1, :]
        next_token_logits = repetition_penalty_processor(generate_tokens, logits)

        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generate_tokens = torch.cat((generate_tokens, next_token), dim=-1)
        if (next_token.view(-1) == stop_speech_token).all():
            break

        # Get embedding for the new token.
        # embed next token
        position_ids = torch.full(
            (input_ids.shape[0], 1),
            i + 1,
            dtype=torch.long,
        )
        ort_embed_tokens_inputs["input_ids"] = next_token.detach().numpy()
        ort_embed_tokens_inputs["position_ids"] = position_ids.detach().numpy()
        next_token_embed = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]

        ## Update values for next generation loop
        attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
        llm_position_ids = llm_position_ids[:, -1:] + 1
        for j, key in enumerate(past_key_values):
            past_key_values[key] = present_key_values[j]

        inputs_embeds = next_token_embed

    speech_tokens = generate_tokens[:, 1:-1]
    speech_tokens = drop_invalid_tokens(speech_tokens)
    speech_tokens = speech_tokens[speech_tokens < 6561]
    speech_tokens = np.concatenate([ref_dict["prompt_token"], speech_tokens.unsqueeze(0)], axis=1)
    token_len = np.array([speech_tokens.shape[1]])
    return speech_tokens, token_len

speech_tokens, token_len = execute_text_to_audio_inference(text, ref_dict)
cond_incoder_input = {
    "speech_tokens": speech_tokens,
    "token_len": token_len,
    "embedding": ref_dict["embedding"],
    "prompt_feat": ref_dict["prompt_feat"],
}
wav = cond_decoder_session.run(None, cond_incoder_input)[0]
watermarker = perth.PerthImplicitWatermarker()
watermarked_wav = watermarker.apply_watermark(np.squeeze(wav, axis=0), sample_rate=S3GEN_SR)
watermarked_wav = torch.from_numpy(watermarked_wav).unsqueeze(0)
ta.save(output_file_name, watermarked_wav, S3GEN_SR)
print(f"{output_file_name} was successfully saved")
