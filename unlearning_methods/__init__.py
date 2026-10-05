"""
Collection of official machine unlearning method implementations:
- Random Label (RL)
- NegGrad+ (Negative Gradient Ascent-Descent)
- SalUn (Saliency-based Unlearning)
- SCRUB (Student-Teacher Distillation Divergence)
- TarUn / UNSIR (Error-Maximizing Impair-Repair)
"""

from .random_label import unlearn_random_label
from .neggrad_plus import unlearn_neggrad_plus
from .salun import unlearn_salun, generate_salun_mask
from .scrub import unlearn_scrub, DistillKL
from .tarun_unsir import unlearn_tarun_unsir

__all__ = [
    "unlearn_random_label",
    "unlearn_neggrad_plus",
    "unlearn_salun",
    "generate_salun_mask",
    "unlearn_scrub",
    "DistillKL",
    "unlearn_tarun_unsir",
]
