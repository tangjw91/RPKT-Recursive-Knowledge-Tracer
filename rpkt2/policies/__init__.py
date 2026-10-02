from .base import Policy
from .baselines import EnumeratePolicy, RandomPolicy, RPKTv1Policy
from .eig import EIGPolicy

REGISTRY = {
    "rpkt_v1": RPKTv1Policy,
    "rpkt_v1_k4": lambda **kw: RPKTv1Policy(max_children=4, **kw),
    "enumerate": EnumeratePolicy,
    "random": RandomPolicy,
    "eig": EIGPolicy,
    "eig_noverify": lambda **kw: EIGPolicy(allow_verify=False, **kw),
}

__all__ = ["Policy", "RPKTv1Policy", "EnumeratePolicy", "RandomPolicy", "EIGPolicy", "REGISTRY"]
