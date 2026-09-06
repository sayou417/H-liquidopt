import pandas as pd
from engine import (
    Coolant, HydraulicGeometry, heat_loads, pod_summary, required_flow_lpm,
    evaluate_coolants, validate_racks, recommended_duty_cdus,
)


def make_racks(n=48):
    rows = []
    for i in range(n):
        rows.append({
            "rack_id": f"R{i+1:03d}",
            "pod": chr(65 + (i // 16)),
            "rack_type": "Compute" if i % 6 else "Support",
            "it_power_kw": 120.0 if i % 6 else 20.0,
            "hcr": 0.85 if i % 6 else 0.0,
            "row": i // 8 + 1,
            "col": i % 8 + 1,
        })
    return pd.DataFrame(rows)


def test_heat_balance():
    racks = make_racks(24)
    out = heat_loads(racks)
    assert ((out["liquid_load_kw"] + out["residual_air_kw"] - out["it_power_kw"]).abs() < 1e-9).all()


def test_flow_positive():
    water = Coolant("Water", 992.2, 4.179, 0.000653)
    assert required_flow_lpm(102.0, water, 10.0) > 0


def test_arbitrary_rack_counts():
    water = Coolant("Water", 992.2, 4.179, 0.000653)
    geom = HydraulicGeometry()
    for n in [12, 24, 48, 64, 96]:
        racks = make_racks(n)
        assert not validate_racks(racks)
        pods = pod_summary(racks)
        results = evaluate_coolants(racks, [water], 10.0, geom)
        assert len(results) == len(pods)
        assert results["pump_kw"].ge(0).all()


def test_validation_duplicate_id():
    racks = make_racks(12)
    racks.loc[1, "rack_id"] = racks.loc[0, "rack_id"]
    assert validate_racks(racks)


def test_cdu_count():
    assert recommended_duty_cdus(4080, 2.0) == 3
