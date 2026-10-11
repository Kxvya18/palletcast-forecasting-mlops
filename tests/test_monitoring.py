"""S7: reading Evidently drift results correctly (p-values vs distances)."""

from palletcast.monitoring.monitor import value_drift


def test_p_value_method_drifts_below_threshold():
    m = {"metric_name": "ValueDrift(column=a,method=K-S p_value,threshold=0.05)", "value": 0.01}
    assert value_drift(m) == ("a", True)
    m["value"] = 0.4
    assert value_drift(m) == ("a", False)


def test_distance_method_drifts_above_threshold():
    m = {
        "metric_id": "ValueDrift(column=b,method=Wasserstein distance (normed),threshold=0.1)",
        "value": 0.3,
    }
    assert value_drift(m) == ("b", True)


def test_other_metrics_are_ignored():
    assert value_drift({"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {}}) == (
        None,
        False,
    )
