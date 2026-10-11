"""S4 promotion gate rule."""

from palletcast.registry.registry import decide

GATE = {"max_wape_increase": 0.003, "max_bias_worsening": 0.01}
PROD = {"wape": 0.120, "bias": -0.020}


def test_equal_or_better_candidate_is_promoted():
    assert decide({"wape": 0.120, "bias": -0.020}, PROD, GATE)["promote"]
    assert decide({"wape": 0.110, "bias": -0.010}, PROD, GATE)["promote"]


def test_small_tolerance_is_allowed():
    assert decide({"wape": 0.1229, "bias": -0.029}, PROD, GATE)["promote"]


def test_clearly_worse_candidate_is_rejected():
    d = decide({"wape": 0.124, "bias": -0.020}, PROD, GATE)
    assert not d["promote"] and not d["wape_ok"]


def test_deeper_under_forecast_is_rejected_even_if_wape_is_fine():
    d = decide({"wape": 0.118, "bias": -0.040}, PROD, GATE)
    assert not d["promote"] and d["wape_ok"] and not d["bias_ok"]
