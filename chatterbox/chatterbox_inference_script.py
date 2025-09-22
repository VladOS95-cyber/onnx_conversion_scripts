# !pip install --upgrade onnxruntime==1.22.1 huggingface_hub==0.34.4 transformers==4.46.3 numpy==2.2.6 tqdm==4.67.1 librosa==0.11.0 soundfile==0.13.1

import onnxruntime

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf

S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562


class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


def run_inference(
    text="The Lord of the Rings is the greatest work of literature.", 
    target_voice_path=None, 
    max_new_tokens = 256,
    exaggeration=0.5, 
    output_dir="converted", 
    output_file_name="output.wav",
    apply_watermark=True,
):

    model_id = "onnx-community/chatterbox-onnx"
    if not target_voice_path:
        target_voice_path = hf_hub_download(repo_id=model_id, filename="default_voice.wav", local_dir=output_dir)

    ## Load model
    speech_encoder_path = hf_hub_download(repo_id=model_id, filename="speech_encoder.onnx", local_dir=output_dir, subfolder='onnx')
    hf_hub_download(repo_id=model_id, filename="speech_encoder.onnx_data", local_dir=output_dir, subfolder='onnx')
    embed_tokens_path = hf_hub_download(repo_id=model_id, filename="embed_tokens.onnx", local_dir=output_dir, subfolder='onnx')
    hf_hub_download(repo_id=model_id, filename="embed_tokens.onnx_data", local_dir=output_dir, subfolder='onnx')
    conditional_decoder_path = hf_hub_download(repo_id=model_id, filename="conditional_decoder.onnx", local_dir=output_dir, subfolder='onnx')
    hf_hub_download(repo_id=model_id, filename="conditional_decoder.onnx_data", local_dir=output_dir, subfolder='onnx')
    language_model_path = hf_hub_download(repo_id=model_id, filename="language_model.onnx", local_dir=output_dir, subfolder='onnx')
    hf_hub_download(repo_id=model_id, filename="language_model.onnx_data", local_dir=output_dir, subfolder='onnx')

    # # Start inferense sessions
    speech_encoder_session = onnxruntime.InferenceSession(speech_encoder_path)
    embed_tokens_session = onnxruntime.InferenceSession(embed_tokens_path)
    llama_with_past_session = onnxruntime.InferenceSession(language_model_path)
    cond_decoder_session = onnxruntime.InferenceSession(conditional_decoder_path)

    def execute_text_to_audio_inference(text):
        print("Start inference script...")

        audio_values, _ = librosa.load(target_voice_path, sr=S3GEN_SR)
        audio_values = audio_values[np.newaxis, :].astype(np.float32)

        ## Prepare input
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        input_ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)

        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1
        )

        ort_embed_tokens_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "exaggeration": np.array([exaggeration], dtype=np.float32)
        }

        ## Instantiate the logits processors.
        repetition_penalty = 1.2
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        num_hidden_layers = 30
        num_key_value_heads = 16
        head_dim = 64

        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.long)

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):

            inputs_embeds = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]
            if i == 0:
                ort_speech_encoder_input = {
                    "audio_values": audio_values,
                }
                cond_emb, prompt_token, ref_x_vector, prompt_feat = speech_encoder_session.run(None, ort_speech_encoder_input)
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)

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

            logits = logits[:, -1, :]
            next_token_logits = repetition_penalty_processor(generate_tokens, logits)

            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)
            if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                break

            # Get embedding for the new token.
            position_ids = np.full(
                (input_ids.shape[0], 1),
                i + 1,
                dtype=np.int64,
            )
            ort_embed_tokens_inputs["input_ids"] = next_token
            ort_embed_tokens_inputs["position_ids"] = position_ids

            ## Update values for next generation loop
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
            llm_position_ids = llm_position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

        speech_tokens = generate_tokens[:, 1:-1]
        speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
        return speech_tokens, ref_x_vector, prompt_feat

    speech_tokens, speaker_embeddings, speaker_features = execute_text_to_audio_inference(text)
    cond_incoder_input = {
        "speech_tokens": speech_tokens,
        "speaker_embeddings": speaker_embeddings,
        "speaker_features": speaker_features,
    }
    wav = cond_decoder_session.run(None, cond_incoder_input)[0]
    wav = np.squeeze(wav, axis=0)

    # Optional: Apply watermark
    if apply_watermark:
        import perth
        watermarker = perth.PerthImplicitWatermarker()
        wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)

    sf.write(output_file_name, wav, S3GEN_SR)
    print(f"{output_file_name} was successfully saved")

if __name__ == "__main__":
    run_inference(
        text="Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
        exaggeration=0.5,
        output_file_name="output.wav",
        apply_watermark=False,
    )
