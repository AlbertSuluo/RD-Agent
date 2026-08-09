import math

import pytest

from rdagent.scenarios.qlib.crypto_factor.config import CryptoFactorPropSetting
from rdagent.scenarios.qlib.crypto_factor.evaluation_adapter import apply_strict_gate


@pytest.fixture
def settings():
    return CryptoFactorPropSetting()


def test_strict_gate_passes_only_above_every_threshold(settings):
    result = apply_strict_gate("good", 0.80001, 0.30001, 0.03001, 0.20001, settings)
    assert result.passed
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("metrics", "failed_name"),
    [
        ((0.8, 0.31, 0.04, 0.3), "PFS"),
        ((0.9, 0.3, 0.04, 0.3), "RRE"),
        ((0.9, 0.31, 0.03, 0.3), "Best_IC"),
        ((0.9, 0.31, 0.04, 0.2), "Best_IR"),
    ],
)
def test_equal_to_threshold_fails(settings, metrics, failed_name):
    result = apply_strict_gate("boundary", *metrics, settings)
    assert not result.passed
    assert any(failed_name in reason and "strictly >" in reason for reason in result.reasons)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, "not-a-number"])
def test_non_finite_or_non_numeric_metrics_fail(settings, bad):
    result = apply_strict_gate("invalid", bad, 0.4, 0.04, 0.3, settings)
    assert not result.passed
    assert "PFS is not finite" in result.reasons
