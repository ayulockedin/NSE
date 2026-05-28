"""Evaluation harness: the scoreboard for patch-selection quality.

Measures how well the latent transition model's ``p_t_latent`` predicts the
sandbox ground truth — calibration (ECE, Brier), discrimination (accuracy,
AUC), and the positive base rate. Run it against the heuristic *before*
training to establish the baseline any trained GNN must beat.
"""
