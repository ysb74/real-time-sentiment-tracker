"""
Fusion logic to aggregate multiple modality sentiments.
Options: majority voting, confidence-weighted, or neural fusion.
"""

import numpy as np

def fuse_modalities(results_dict):
    """
    Given:
        results_dict = {
            "text": {"label": "positive", "confidence": 0.8},
            "image": {"label": "neutral", "confidence": 0.7},
            "audio": {"label": "negative", "confidence": 0.9}
        }
    Returns:
        {"label": ..., "breakdown": results_dict}
    """
    votes = {}
    for modality, res in results_dict.items():
        label = res["label"]
        conf = res.get("confidence", 1.0)
        votes[label] = votes.get(label, 0.0) + conf
    final_label = max(votes, key=votes.get)
    return {"label": final_label, "breakdown": results_dict}
