# !pip install --upgrade torch==2.6.0 torchaudio==2.6.0 numpy==2.2.6 librosa==0.11.0 onnx==1.18.0 onnxslim==0.1.59

import torch

import torchaudio as ta

import librosa
from tqdm import tqdm

from chatterbox_to_onnx_conversion_script import (
    SafeDenseLayer, 
    PrepareConditionalsModel, 
    InputsEmbeds, 
    ConditionalDecoder, 
    S3GEN_SR,
    EXAGGERATION_TOKEN,
    START_SPEECH_TOKEN,
    STOP_SPEECH_TOKEN)


@torch.no_grad()
def export_model_to_onnx(
    export_prepare_conditions=False, 
    export_cond_decoder=False, 
    audio_prompt_path=None, 
    output_export_dir=None, 
    output_file_name="output.wav", 
    device="cpu"):
    # pip install chatterbox-tts==0.1.3 or 0.1.4
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if output_export_dir:
        import os
        os.makedirs(output_export_dir, exist_ok=True)
    chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

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
            f"{output_export_dir}/multi_lang_embed_tokens.onnx",
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
        print(f"✅ Embedding Tokens ONNX export is completed. Model saved as 'multi_lang_embed_tokens.onnx'")

        dummy_audio_values = torch.randn(1, 312936)
        torch.onnx.export(
            prepare_conditionals,
            (dummy_audio_values, ),
            f"{output_export_dir}/multi_lang_speech_encoder.onnx",
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
        print(f"✅ Speech Encoder ONNX export is completed. Model saved as 'multi_lang_speech_encoder.onnx'")


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
            f"{output_export_dir}/multi_lang_conditional_decoder.onnx",
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
        print(f"✅ Conditional decoder ONNX export is completed. Model saved as 'multi_lang_conditional_decoder.onnx'")

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
