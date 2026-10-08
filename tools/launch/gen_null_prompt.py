"""Regenerate the cached null-prompt umT5 embedding (null_prompt_umt5.pt).

Reproduces VideoX-Fun's _get_t5_prompt_embeds("") semantics: tokenize the empty
string (-> a single </s> token), encode with the Wan umT5-xxl text encoder, trim
to the real sequence length. Deterministic given the same T5 weights.
"""
import os

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from videox_fun.models.wan_text_encoder import WanT5EncoderModel

# The official Wan T5 .pth is legacy-tar serialized; torch>=2.6 defaults
# weights_only=True which rejects it. The file comes from the official HF repo.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

WAN_DIR = os.environ["WAN_DIR"]
VIDEOX_HOME = os.environ["VIDEOX_HOME"]
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = OmegaConf.load(os.path.join(VIDEOX_HOME, "config/wan2.1/wan_civitai.yaml"))
tok_sub = cfg["text_encoder_kwargs"].get("tokenizer_subpath", "google/umt5-xxl")
te_sub = cfg["text_encoder_kwargs"].get("text_encoder_subpath", "models_t5_umt5-xxl-enc-bf16.pth")

tokenizer = AutoTokenizer.from_pretrained(os.path.join(WAN_DIR, tok_sub))
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(WAN_DIR, te_sub),
    additional_kwargs=OmegaConf.to_container(cfg["text_encoder_kwargs"]),
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
).eval().to(device)

ti = tokenizer(
    [""],
    padding="max_length",
    max_length=512,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt",
)
seq_len = ti.attention_mask.gt(0).sum(dim=1).long()[0].item()
with torch.no_grad():
    emb = text_encoder(ti.input_ids.to(device), attention_mask=ti.attention_mask.to(device))[0]
null = emb[0, :seq_len].cpu()
out = os.path.join(WAN_DIR, "null_prompt_umt5.pt")
torch.save(null, out)
print(f"saved {out}: shape={tuple(null.shape)} dtype={null.dtype} seq_len={seq_len}")
