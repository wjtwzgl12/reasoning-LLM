"""§9A.3 PRM inference wrapper — loads LoRA adapter + scalar head and
exposes a `PRMScorer.score(text) → float ∈ [0, 1]` contract compatible with
`weaver_signals.extract_w1_prm(prm_model=...)`.

Training recipe (from colab/9A3_prm_training.ipynb):
  - Backbone: Qwen/Qwen2.5-1.5B-Instruct, frozen, bf16
  - LoRA r=16, alpha=32, dropout=0.05, target=q/k/v/o_proj
  - Head: Linear(hidden_size → 1) in fp32
  - Forward: last hidden state at last non-pad position → head → sigmoid
  - Gate (§9A.3): eval AUROC ≥ 0.70 — PASSed at 0.955

Artifacts (default layout):
  models/pvgap_prm_v1/prm_v1_lora/           # peft adapter dir
  models/pvgap_prm_v1/prm_v1_head.pt         # torch.save(head.state_dict())
  models/pvgap_prm_v1/prm_v1_metrics.json    # training history
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
_DEFAULT_ROOT = HERE.parent / "models" / "pvgap_prm_v1"
_DEFAULT_BACKBONE = "Qwen/Qwen2.5-1.5B-Instruct"
_MAX_LEN = 2048


class PRMScorer:
    """Load once, score many times. Thread-unsafe (single CUDA/CPU session)."""

    def __init__(self,
                 root: str | Path = _DEFAULT_ROOT,
                 backbone_id: str = _DEFAULT_BACKBONE,
                 device: str | None = None,
                 dtype: str = "bf16"):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        from peft import PeftModel

        root = Path(root)
        adapter_dir = root / "prm_v1_lora"
        head_path   = root / "prm_v1_head.pt"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"LoRA adapter dir missing: {adapter_dir}")
        if not head_path.is_file():
            raise FileNotFoundError(f"head ckpt missing: {head_path}")

        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[dtype]
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_id,
                                                       trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModel.from_pretrained(backbone_id,
                                         trust_remote_code=True,
                                         torch_dtype=torch_dtype)
        base = base.to(self.device)
        base.eval()
        backbone = PeftModel.from_pretrained(base, str(adapter_dir))
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone

        hidden = backbone.config.hidden_size
        head = nn.Linear(hidden, 1).to(self.device).to(torch.float32)
        head.load_state_dict(torch.load(head_path, map_location=self.device))
        head.eval()
        for p in head.parameters():
            p.requires_grad = False
        self.head = head

        # Bookkeeping: training used a single logit → BCE-with-logits, so
        # inference score = sigmoid(logit).
        metrics_path = root / "prm_v1_metrics.json"
        self.metrics = (json.loads(metrics_path.read_text(encoding="utf-8"))
                        if metrics_path.is_file() else {})

    # ------------------------------------------------------------------

    def _forward_logits(self, texts: list[str]) -> "np.ndarray":
        import torch
        enc = self.tokenizer(texts, truncation=True, max_length=_MAX_LEN,
                             padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.backbone(**enc)
            h = out.last_hidden_state                                  # [B, T, H]
            seq_len = enc["attention_mask"].sum(dim=1) - 1             # last non-pad idx
            h_last = h[torch.arange(h.size(0), device=self.device), seq_len]
            logits = self.head(h_last.to(torch.float32)).squeeze(-1)   # [B]
        return logits.detach().cpu().numpy()

    def score_batch(self, texts: Iterable[str]) -> list[float]:
        texts = list(texts)
        if not texts:
            return []
        logits = self._forward_logits(texts)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return [float(p) for p in probs]

    def score(self, text: str) -> float:
        """weaver_signals.extract_w1_prm contract: float in [0,1]."""
        if not text or not text.strip():
            return 0.5
        return self.score_batch([text])[0]


# ─────────────────────── self-test / smoke CLI ─────────────────────────


def _cli():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_DEFAULT_ROOT))
    ap.add_argument("--backbone", default=_DEFAULT_BACKBONE)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    print(f"[prm_scorer] loading from {args.root} …")
    t0 = time.time()
    scorer = PRMScorer(root=args.root, backbone_id=args.backbone,
                       device=args.device, dtype=args.dtype)
    print(f"  loaded in {time.time()-t0:.1f}s  | metrics: "
          f"final_auroc={scorer.metrics.get('final_auroc')}")

    probes = [
        # Should score HIGH: physically sensible stepwise EIS reasoning
        ("Step 1: The high-frequency semicircle diameter indicates the "
         "charge-transfer resistance R_ct ≈ 25 mΩ.\n\nStep 2: The 45-degree "
         "Warburg tail at low frequency confirms semi-infinite diffusion."),
        # Should score LOW: unrelated / wrong
        ("Step 1: The cell voltage equals 3.7 V under open circuit.\n\n"
         "Step 2: Therefore the SEI layer must be growing."),
        # Neutral baseline
        ("Step 1: We measured EIS from 0.1 Hz to 1 kHz at SoC 50%."),
    ]
    t0 = time.time()
    scores = scorer.score_batch(probes)
    print(f"  {len(probes)} probes in {time.time()-t0:.2f}s")
    for s, t in zip(scores, probes):
        head = t.replace("\n", " ")[:70]
        print(f"    w1={s:.3f}  | {head}…")


if __name__ == "__main__":
    _cli()
