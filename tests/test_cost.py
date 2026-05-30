"""Tests for the cost model + cost-aware acquisition (Phase 10.2)."""

import math

import pytest

from nse.config import SETTINGS
from nse.orchestrator.cost import (
    CostLedger,
    CostModel,
    acquisition_score,
    binary_entropy,
    info_gain,
    rank_by_acquisition,
)


def test_binary_entropy_peaks_at_half():
    assert binary_entropy(0.5) == pytest.approx(1.0)        # 1 bit, maximal
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) > binary_entropy(0.9) > binary_entropy(0.99)
    assert binary_entropy(0.1) == pytest.approx(binary_entropy(0.9))  # symmetric


def test_cost_model_pricing():
    cm = CostModel(gnn_cost=0.0, llm_cost_per_1k_tokens=0.05, sandbox_cost_per_run=1.0)
    assert cm.gnn(5) == 0.0
    assert cm.llm(2000) == pytest.approx(0.1)
    assert cm.sandbox(3) == pytest.approx(3.0)


def test_cost_ledger_totals():
    led = CostLedger()
    led.add_gnn(0.01)
    led.add_llm(0.2)
    led.add_sandbox(1.0)
    assert led.total == pytest.approx(1.21)
    assert led.as_dict()["total"] == pytest.approx(1.21)


def test_info_gain_combines_aleatoric_and_epistemic():
    # At p=0.5 the aleatoric term is maximal (1 bit); u adds epistemic on top.
    base = info_gain(0.5, u=0.0)
    with_u = info_gain(0.5, u=SETTINGS.hp.max_variance, gamma=1.0)
    assert base == pytest.approx(1.0)
    assert with_u == pytest.approx(2.0)            # +normalize_uncertainty(max)=+1
    # A confident prediction is less informative to sandbox than a 50/50 one.
    assert info_gain(0.95, u=0.0) < info_gain(0.5, u=0.0)


def test_acquisition_is_info_per_cost():
    assert acquisition_score(0.5, 0.0, cost=2.0) == pytest.approx(0.5)
    assert math.isinf(acquisition_score(0.5, 0.0, cost=0.0))  # free run


def test_rank_by_acquisition_prioritizes_the_uncertain():
    items = [("certain", 0.98, 0.0), ("coinflip", 0.5, 0.0), ("likely", 0.8, 0.0)]
    ranked = rank_by_acquisition(items, cost=1.0)
    assert ranked[0] == "coinflip"      # most uncertain -> most worth a sandbox run
    assert ranked[-1] == "certain"
