import onnxruntime
import numpy as np
from huggingface_hub import hf_hub_download
import torchaudio as ta
import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from tqdm import tqdm
import perth
import librosa
from transformers import AutoTokenizer
import torch.nn.functional as F

SPACE = "[SPACE]"
S3GEN_SR = 24000
# Sampling rate of the inputs to S3TokenizerV2
EXAGGERATION_TOKEN = 6563
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562

def run_inference(
        text="The Lord of the Rings is the greatest work of literature.", 
        target_voice_path=None, 
        max_new_tokens = 128,
        exaggeration=0.5, 
        output_dir="converted", 
        output_file_name="output.wav" ):

    model_id = "onnx-community/chatterbox-onnx"
    target_voice_path = hf_hub_download(repo_id=model_id, filename="default_voice.wav", local_dir=output_dir) if not target_voice_path else target_voice_path

    ## Load model
    speech_encoder_path = hf_hub_download(repo_id=model_id, filename="speech_encoder.onnx", local_dir=output_dir, subfolder='onnx')
    language_model_path = hf_hub_download(repo_id=model_id, filename="language_model.onnx", local_dir=output_dir, subfolder='onnx')
    embed_tokens_path = hf_hub_download(repo_id=model_id, filename="embed_tokens.onnx", local_dir=output_dir, subfolder='onnx')
    hf_hub_download(repo_id=model_id, filename="language_model.onnx_data", local_dir=output_dir, subfolder='onnx')
    conditional_decoder_path = hf_hub_download(repo_id=model_id, filename="conditional_decoder.onnx", local_dir=output_dir, subfolder='onnx')

    ## Start inferense sessions
    speech_encoder_session = onnxruntime.InferenceSession(speech_encoder_path)
    embed_tokens_session = onnxruntime.InferenceSession(embed_tokens_path)
    llama_with_past_session = onnxruntime.InferenceSession(language_model_path)
    cond_decoder_session = onnxruntime.InferenceSession(conditional_decoder_path)

    def execute_text_to_audio_inference(text):
        print("Start inference script...")

        audio_values, _ = librosa.load(target_voice_path, sr=S3GEN_SR)
        audio_values = torch.from_numpy(audio_values).unsqueeze(0)

        ## Prepare input
        text = text.replace(' ', SPACE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text_tokens_ids = tokenizer(text)["input_ids"]
        text_tokens_ids = torch.IntTensor(text_tokens_ids).unsqueeze(0)
        input_ids = F.pad(text_tokens_ids, (0, 1), value=START_SPEECH_TOKEN)
        input_ids = F.pad(input_ids, (0, 1), value=START_SPEECH_TOKEN)
        input_ids = F.pad(input_ids, (1, 0), value=EXAGGERATION_TOKEN)
        position_ids = torch.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            torch.arange(input_ids.shape[1]).unsqueeze(0) - 1
        )

        ort_embed_tokens_inputs = {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "position_ids": position_ids.cpu().numpy(),
            "exaggeration": np.array([exaggeration], dtype=np.float32)
        }

        ## Instantiate the logits processors.
        repetition_penalty=1.2
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        num_hidden_layers = 30
        num_key_value_heads = 16
        head_dim = 64

        generate_tokens = torch.tensor([[START_SPEECH_TOKEN]], dtype=torch.long)

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):

            inputs_embeds = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]
            if i == 0:
                ort_speech_encoder_input = {
                    "audio_values": audio_values.detach().cpu().numpy().astype(np.float32),
                }
                cond_emb, prompt_token, ref_x_vector, prompt_feat = speech_encoder_session.run(None, ort_speech_encoder_input)
                inputs_embeds = np.concat((cond_emb, inputs_embeds), axis=1)

                ## Prepare llm inputs
                batch_size, seq_len, _ = inputs_embeds.shape
                past_key_values = {
                    f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
                    for layer in range(num_hidden_layers)
                    for kv in ("key", "value")
                }
                attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
                llm_position_ids = np.cumsum(attention_mask, axis=1, dtype=np.int64) - 1

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
            if (next_token.view(-1) == STOP_SPEECH_TOKEN).all():
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

            ## Update values for next generation loop
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
            llm_position_ids = llm_position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]


        speech_tokens = generate_tokens[:, 1:-1]
        speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
        token_len = np.array([speech_tokens.shape[1]])
        return speech_tokens, token_len, ref_x_vector, prompt_feat

    speech_tokens, token_len, ref_x_vector, prompt_feat = execute_text_to_audio_inference(text)
    cond_incoder_input = {
        "speech_tokens": speech_tokens,
        "token_len": token_len,
        "embedding": ref_x_vector,
        "prompt_feat": prompt_feat,
    }
    wav = cond_decoder_session.run(None, cond_incoder_input)[0]
    watermarker = perth.PerthImplicitWatermarker()
    watermarked_wav = watermarker.apply_watermark(np.squeeze(wav, axis=0), sample_rate=S3GEN_SR)
    watermarked_wav = torch.from_numpy(watermarked_wav).unsqueeze(0)
    ta.save(output_file_name, watermarked_wav, S3GEN_SR)
    print(f"{output_file_name} was successfully saved")

if __name__ == "__main__":
    run_inference()