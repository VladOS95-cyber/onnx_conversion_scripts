# # !pip install --upgrade onnx==1.17.0 onnxruntime==1.20.1 onnxslim==0.1.48 transformers==4.48.3 chatterbox-tts==0.1.2

import torch
from torch import nn
import torch.nn.functional as F
from chatterbox.tts import ChatterboxTTS, T3Cond
import onnxslim
import os

SPEECH_VOCAB_SIZE = 6561
SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1


def create_dummy_inputs(hp, batch_size=2, text_len=40):
    speaker_emb = torch.randn(batch_size, hp.speaker_embed_size)
    cond_prompt_speech_tokens = torch.randint(
        0, hp.speech_tokens_dict_size, (batch_size, hp.speech_cond_prompt_len)
    )
    emotion_adv = torch.tensor([0.5] * batch_size).unsqueeze(1)
    text_tokens_ids = torch.randint(
        0, hp.text_tokens_dict_size, (batch_size, text_len)
    )
    sot = hp.start_text_token
    eot = hp.stop_text_token
    text_tokens_ids = F.pad(text_tokens_ids, (1, 0), value=sot)
    text_tokens_ids = F.pad(text_tokens_ids, (0, 1), value=eot)
    speech_input_ids = hp.start_speech_token * torch.ones_like(text_tokens_ids[:, :1])
    return speaker_emb, cond_prompt_speech_tokens, emotion_adv, text_tokens_ids, speech_input_ids

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
        self.register_buffer("bos_token", torch.tensor([[self.hp.start_speech_token]], dtype=torch.long))

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
        cond_emb =  self.cond_enc(t3_cond)
        text_emb = self.text_emb(text_tokens_ids)
        speech_emb = self.speech_emb(speech_input_ids)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens_ids)
            speech_emb = speech_emb + self.speech_pos_emb(speech_input_ids)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        inputs_embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
        bos_embed = self.speech_emb(self.bos_token)
        idx = torch.tensor(0)
        idx = torch.atleast_2d(idx)
        idx =  self.speech_pos_emb(idx)
        bos_embed = bos_embed + idx
        bos_embed = torch.cat([bos_embed, bos_embed])
        inputs_embeds = torch.cat([inputs_embeds, bos_embed], dim=1)
        return inputs_embeds, len_cond

speaker_emb, cond_prompt_speech_tokens, emotion_adv, text_tokens_ids, speech_input_ids = create_dummy_inputs(config)
embedder_model = SpeechEncoder()
torch.onnx.export(
    embedder_model,
    (speaker_emb, cond_prompt_speech_tokens, emotion_adv, text_tokens_ids, speech_input_ids),
    f"{output_dir}/speech_encoder.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
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
)
print(f"✅ Speech Encoder ONNX export is completed. Model saved as 'speech_encoder.onnx'")

# 2. Export LLM Backbone (Llama3 from repo vladislavbro/llama_backbone_0.5) using https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py
# Before export, we replace embed_tokens and lm_head of the LLM backbone model
# because it does not specifically use lllm tokens

# model.t3.tfmr.embed_tokens.weight = torch.nn.Parameter(model.t3.speech_head.weight.contiguous())
# model.t3.cfg.vocab_size = model.t3.speech_head.out_features
# model_to_exp = LlamaForCausalLM(model.t3.cfg)
# model_to_exp.lm_head = model.t3.speech_head
# model_to_exp.model = model.t3.tfmr

hidden_states = torch.randn(1, text_tokens_ids.shape[1], config.n_channels)
head_out = model.t3.speech_head(hidden_states)

speech_tokens = torch.randint(
        0, config.text_tokens_dict_size, (head_out.shape[1],)
    )

# 3. Export Flow inference wrapper
class FlowInferenceWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.ref_dict = {k: v.detach() if v is not None else None for k, v in model.conds.gen.items()}
        self.flow = model.s3gen.flow
        self.input_size = model.s3gen.flow.input_size
        self.output_size = model.s3gen.flow.output_size
        self.decoder_conf = model.s3gen.flow.decoder_conf
        self.mel_feat_conf = model.s3gen.flow.mel_feat_conf
        self.vocab_size = model.s3gen.flow.vocab_size
        self.output_type = model.s3gen.flow.output_type
        self.input_frame_rate = model.s3gen.flow.input_frame_rate
        self.input_embedding = model.s3gen.flow.input_embedding
        self.spk_embed_affine_layer = model.s3gen.flow.spk_embed_affine_layer
        self.encoder = model.s3gen.flow.encoder
        self.encoder_proj = model.s3gen.flow.encoder_proj
        self.only_mask_loss = model.s3gen.flow.only_mask_loss
        self.token_mel_ratio = model.s3gen.flow.token_mel_ratio
        self.pre_lookahead_len = model.s3gen.flow.pre_lookahead_len
    
    def make_pad_mask(self, lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
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

    def forward(self, speech_tokens):
        prompt_token = self.ref_dict["prompt_token"]
        prompt_token_len = self.ref_dict["prompt_token_len"]
        embedding = self.ref_dict["embedding"]
        prompt_feat = self.ref_dict["prompt_feat"]
        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        # assert speech_tokens.shape[0] == 1, "only batch size of one allowed for now"
        speech_token_lens = torch.LongTensor([speech_tokens.size(1)])

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        speech_tokens, token_len = torch.concat([prompt_token, speech_tokens], dim=1), prompt_token_len + speech_token_lens
        mask = (~self.make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        speech_tokens = self.input_embedding(torch.clamp(speech_tokens, min=0)) * mask

        # text encode
        text_encoded, _ = self.encoder(speech_tokens, token_len)
        mel_len1, mel_len2 = prompt_feat.shape[1], text_encoded.shape[1] - prompt_feat.shape[1]
        text_encoded = self.encoder_proj(text_encoded)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size] ).to(text_encoded.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        total_len = torch.tensor([mel_len1 + mel_len2])
        mask = (~self.make_pad_mask(total_len)).to(text_encoded)
        mel_len1 = torch.tensor([mel_len1], dtype=torch.int64)
        mu = text_encoded.transpose(1, 2).contiguous()
        mask = mask.unsqueeze(1)
        spks = embedding
        cond = conds
        return mel_len1, mu, mask, spks, cond

flow_inference = FlowInferenceWrapper()
torch.onnx.export(
    flow_inference,
    (speech_tokens),
    f"{output_dir}/flow_inference.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["speech_tokens"],
    output_names=["mel_len1", "mu", "mask", "spks", "cond"],
    dynamic_axes={
        "speech_tokens": {0: "sequence_len"},
        "mu": {0: "batch_size", 2: "output_time"},
        "mask": {0: "batch_size", 2: "output_time"},
        "spks": {0: "batch_size"},
        "cond": {0: "batch_size", 2: "output_time"},
    }
)
print(f"✅ Flow inference ONNX export is completed. Model saved as 'flow_inference.onnx'")

# 4. Export Conditional Decoder
class ConditionalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = model.s3gen.flow.decoder.estimator
        self.t_scheduler = model.s3gen.flow.decoder.t_scheduler
        self.inference_cfg_rate = model.s3gen.flow.decoder.inference_cfg_rate
        self.n_timesteps = 10

    def forward(self, mel_len1, mu, mask, spks, cond, rand_noise):
        temperature = 1.0
        B, _, T = mu.shape
        x = rand_noise[:, :, :T].to(mu.device).to(mu.dtype) * temperature

        t_span = torch.linspace(0, 1, self.n_timesteps+1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        dt_all = t_span[1:] - t_span[:-1]
        t = t_span[0:1]

        x_in = torch.zeros([2, 80, T], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, T], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, T], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, T], device=x.device, dtype=x.dtype)

        for i in range(self.n_timesteps ):
            dt = dt_all[i:i+1]  # keep shape
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t
            spks_in[0] = spks
            cond_in[0] = cond

            dphi_dt = model.s3gen.flow.decoder.forward_estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [B, B], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt

            x = x + dt * dphi_dt
            t = t + dt
        feat = x.float()
        output_mels = feat[:, :, mel_len1.squeeze():]
        output_mels = output_mels.float()
        return output_mels

cond_decoder = ConditionalDecoder()
mel_len1, mu, mask, spks, cond = flow_inference(speech_tokens)
rand_noise = model.s3gen.flow.decoder.rand_noise
torch.onnx.export(
    cond_decoder,
    (mel_len1, mu, mask, spks, cond, rand_noise),
    f"{output_dir}/conditional_decoder.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["mel_len1", "mu", "mask", "spks", "cond", "rand_noise"],
    output_names=["output_mels"],
    dynamic_axes={
        'mu': {0: 'batch_size', 2: 'input_time'},
        'mask': {0: 'batch_size', 2: 'input_time'},
        'spks': {0: 'batch_size'},
        'cond': {0: 'batch_size', 2: 'input_time'},
        'output_mels': {0: 'batch_size', 2: 'output_time'}
    }
)
print(f"✅ Conditional decoder ONNX export is completed. Model saved as 'conditional_decoder.onnx'")

# 5. Post-processing
for f in os.listdir(output_dir):
    p = os.path.join(output_dir, f)
    onnxslim.slim(p, p)