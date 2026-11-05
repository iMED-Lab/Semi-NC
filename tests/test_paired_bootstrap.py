from scripts.paired_bootstrap import paired_patient_bootstrap


def _report(values):
    return {
        "records": [
            {
                "image": f"image_{index}",
                "patient_id": patient,
                "class_index": 0,
                "metric": "dice",
                "value": value,
            }
            for index, (patient, value) in enumerate(values)
        ]
    }


def test_paired_bootstrap_resamples_whole_patient_clusters():
    ours = _report([("P1", 0.9), ("P1", 0.8), ("P2", 0.7)])
    baseline = _report([("P1", 0.7), ("P1", 0.6), ("P2", 0.5)])
    result = paired_patient_bootstrap(ours, baseline, samples=100, seed=1)
    metric = result["class_0/dice"]
    assert metric["patient_count"] == 2
    assert abs(metric["observed_improvement"] - 0.2) < 1e-12
    assert metric["p_value_two_sided"] == 0.0


def test_paired_bootstrap_rejects_missing_patient_ids():
    ours = _report([("", 0.9)])
    baseline = _report([("", 0.8)])
    try:
        paired_patient_bootstrap(ours, baseline, samples=10)
    except ValueError as error:
        assert "patient_id" in str(error)
    else:
        raise AssertionError("Missing patient clusters must not become image clusters")
