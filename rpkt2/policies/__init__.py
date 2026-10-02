from .base import Policy
from .baselines import EnumeratePolicy, RandomPolicy, RPKTv1Policy
from .eig import EIGPolicy

def _rl(model_path: str = "models/rl_policy.pt", **kw):
    """Lazy import so the package works without PyTorch installed."""
    import torch

    from .rl import GNNPolicyNet, RLPolicy
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    net = GNNPolicyNet()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    use_belief = not ckpt["args"].get("no_belief", False)
    return RLPolicy(net=net, use_belief=use_belief, **kw)


REGISTRY = {
    "rpkt_v1": RPKTv1Policy,
    "rpkt_v1_k4": lambda **kw: RPKTv1Policy(max_children=4, **kw),
    "enumerate": EnumeratePolicy,
    "random": RandomPolicy,
    "eig": EIGPolicy,
    "eig_noverify": lambda **kw: EIGPolicy(allow_verify=False, **kw),
    "rl": _rl,
    "rl_nobelief": lambda **kw: _rl(model_path="models/rl_policy_nobelief.pt", **kw),
}

__all__ = ["Policy", "RPKTv1Policy", "EnumeratePolicy", "RandomPolicy", "EIGPolicy", "REGISTRY"]
