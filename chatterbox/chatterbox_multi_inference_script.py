# !pip install --upgrade onnxruntime==1.22.1 huggingface_hub==0.34.4 transformers==4.46.3 numpy==2.2.6 tqdm==4.67.1 librosa==0.11.0 soundfile==0.13.1 perth==1.0.0
# for Chinese, Japanese additionally pip install pkuseg==0.0.25 pykakasi==2.3.0

import onnxruntime

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
from unicodedata import category
import json

S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


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


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""
    
    def __init__(self):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping()
        self._init_segmenter()
    
    def _load_cangjie_mapping(self):
        """Load Cangjie mapping from HuggingFace model repository."""        
        try:
            cangjie_file = hf_hub_download(
                repo_id="onnx-community/chatterbox-multilingual-ONNX",
                filename="Cangjie5_TC.json",
            )
            
            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)
                    
        except Exception as e:
            print(f"Could not load Cangjie mapping: {e}")
    
    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            print("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None
    
    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)
    

    
    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text
        
        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = []
                for c in cangjie:
                    code.append(f"[cj_{c}]")
                code.append("[cj_.]")
                code = "".join(code)
                output.append(code)
            else:
                output.append(t)
        return "".join(output)


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese text normalization: converts kanji to hiragana; katakana remains the same."""
    global _kakasi
    
    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()
        
        result = _kakasi.convert(text)
        out = []
        
        for r in result:
            inp = r['orig']
            hira = r["hira"]

            # Any kanji in the phrase
            if any([is_kanji(c) for c in inp]):
                if hira and hira[0] in ["は", "へ"]:  # Safety check for empty hira
                    hira = " " + hira
                out.append(hira)

            # All katakana
            elif all([is_katakana(c) for c in inp]) if inp else False:  # Safety check for empty inp
                out.append(r['orig'])

            else:
                out.append(inp)
        
        normalized_text = "".join(out)
        
        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', normalized_text)
        
        return normalized_text
        
    except ImportError:
        print("pykakasi not available - Japanese text processing skipped")
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta
    
    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            _dicta = Dicta()
        
        return _dicta.add_diacritics(text)
        
    except ImportError:
        print("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        print(f"Hebrew diacritization failed: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""
    
    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        
        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        
        return initial + medial + final
    
    # Decompose syllables and normalize punctuation
    result = ''.join(decompose_hangul(char) for char in text)    
    return result.strip()


def prepare_language(txt, language_id):
    # Language-specific text processing
    cangjie_converter = ChineseCangjieConverter()
    if language_id == 'zh':
        txt = cangjie_converter(txt)
    elif language_id == 'ja':
        txt = hiragana_normalize(txt)
    elif language_id == 'he':
        txt = add_hebrew_diacritics(txt)
    elif language_id == 'ko':
        txt = korean_normalize(txt)
    
    # Prepend language token
    if language_id:
        txt = f"[{language_id.lower()}]{txt}"
    return txt


def run_inference(
    text="The Lord of the Rings is the greatest work of literature.",
    language_id="en", 
    target_voice_path=None, 
    max_new_tokens = 256,
    exaggeration=0.5, 
    output_dir="converted", 
    output_file_name="output.wav",
    apply_watermark=True,
):
    # Validate language_id
    if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(
            f"Unsupported language_id '{language_id}'. "
            f"Supported languages: {supported_langs}"
        )
    model_id = "onnx-community/chatterbox-multilingual-ONNX"
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
        text = prepare_language(text, language_id)
        input_ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)

        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1
        )

        ort_embed_tokens_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids.astype(np.int64),
            "exaggeration": np.array([exaggeration], dtype=np.float32)
        }

        ## Instantiate the logits processors.
        repetition_penalty = 1.2
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        num_hidden_layers = 30
        num_key_value_heads = 16
        head_dim = 64

        generate_tokens = np.array([[START_SPEECH_TOKEN]])

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
            logits, *present_key_values = llama_with_past_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
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
        target_voice_path="Sophia.wav",
        text="Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues.",
        language_id="fr",
        exaggeration=0.5,
        output_file_name="output.wav",
        apply_watermark=False,
    )
