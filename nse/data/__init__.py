"""Data engine: synthetic mutation generation + labeled dataset construction.

The latent transition model needs supervised (features -> tests_passed) data.
We obtain it by mutating known-good repos and running every mutant through the
*verified* sandbox so the label is ground truth, never an assumption.
"""
