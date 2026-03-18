import numpy as np

DELAY_DELTAS = [f"{delta:.1f}" for delta in np.arange(0.0, 0.61, 0.1)]

HANDCRAFTED_METHOD_ALLOWLIST = [
    "avg_token_prob",
    "avg_token_entropy",
    "max_token_prob",
    "max_token_entropy",
    
    "total_var",
    "pos_var",
    "rot_var",
    "gripper_var",
    "entropy_linkage0.01",
    "entropy_linkage0.05",
    "stac_mmd",
    "stac_single",
]
