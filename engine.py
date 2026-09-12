from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


REQUIRED_RACK_COLUMNS = {"rack_id", "pod", "rack_type", "it_power_kw", "hcr"}
OPTIONAL_LAYOUT_COLUMNS = {"row", "col"}


@dataclass
class Coolant:
    name: str
    rho_kg_m3: float
    cp_kj_kgk: float
    mu_pa_s: float
    status: str = "Validated input"


@dataclass
class HydraulicGeometry:
    common_length_m: float = 20.0
    common_diameter_m: float = 0.2027
    row_length_m: float = 12.0
    row_diameter_m: float = 0.1541
    branch_length_m: float = 6.0
    branch_diameter_m: float = 0.0525
    roughness_m: float = 1.5e-5
    common_minor_k: float = 4.0
    branch_minor_k: float = 10.0
    pump_efficiency: float = 0.75
    motor_efficiency: float = 0.92
    rack_dp_reference_kpa: float = 120.0


def validate_racks(racks: pd.DataFrame) -> list[str]:
    """Return human-readable validation errors. Empty list means usable input."""
    errors: list[str] = []
    missing = REQUIRED_RACK_COLUMNS - set(racks.columns)
    if missing:
        return [f"Missing required columns: {', '.join(sorted(missing))}"]
    if racks.empty:
        errors.append("Rack table is empty.")
        return errors
    if racks["rack_id"].isna().any() or (racks["rack_id"].astype(str).str.strip() == "").any():
        errors.append("rack_id cannot be blank.")
    if racks["rack_id"].astype(str).duplicated().any():
        dup = racks.loc[racks["rack_id"].astype(str).duplicated(), "rack_id"].astype(str).tolist()
        errors.append(f"Duplicate rack_id detected: {', '.join(dup[:5])}")
    if racks["pod"].isna().any() or (racks["pod"].astype(str).str.strip() == "").any():
        errors.append("pod cannot be blank.")
    for col in ["it_power_kw", "hcr"]:
        converted = pd.to_numeric(racks[col], errors="coerce")
        if converted.isna().any():
            errors.append(f"{col} must be numeric for every rack.")
    if not errors:
        if (pd.to_numeric(racks["it_power_kw"]) < 0).any():
            errors.append("it_power_kw cannot be negative.")
        hcr = pd.to_numeric(racks["hcr"])
        if ((hcr < 0) | (hcr > 1)).any():
            errors.append("hcr must be between 0 and 1.")
    if {"row", "col"}.issubset(racks.columns):
        if racks[["row", "col"]].isna().any().any():
            errors.append("row/col cannot be blank when layout columns are supplied.")
        elif racks[["row", "col"]].duplicated().any():
            errors.append("Two racks occupy the same row/col position.")
    return errors


def heat_loads(racks: pd.DataFrame) -> pd.DataFrame:
    errors = validate_racks(racks)
    if errors:
        raise ValueError(" | ".join(errors))
    out = racks.copy()
    out["it_power_kw"] = pd.to_numeric(out["it_power_kw"]).astype(float)
    out["hcr"] = pd.to_numeric(out["hcr"]).astype(float)
    out["liquid_load_kw"] = out["it_power_kw"] * out["hcr"]
    out["residual_air_kw"] = out["it_power_kw"] - out["liquid_load_kw"]
    out["liquid_cooled"] = out["liquid_load_kw"] > 1e-9
    return out


def pod_summary(racks: pd.DataFrame) -> pd.DataFrame:
    x = heat_loads(racks)
    return (
        x.groupby("pod", as_index=False, sort=False)
        .agg(
            rack_positions=("rack_id", "count"),
            liquid_racks=("liquid_cooled", "sum"),
            it_load_kw=("it_power_kw", "sum"),
            liquid_load_kw=("liquid_load_kw", "sum"),
            residual_air_kw=("residual_air_kw", "sum"),
        )
    )


def required_flow_lpm(q_kw: float, coolant: Coolant, delta_t_k: float) -> float:
    if delta_t_k <= 0 or coolant.cp_kj_kgk <= 0 or coolant.rho_kg_m3 <= 0:
        raise ValueError("delta-T, Cp and density must be positive")
    if q_kw < 0:
        raise ValueError("Heat load cannot be negative")
    mdot_kg_s = q_kw / (coolant.cp_kj_kgk * delta_t_k)
    return mdot_kg_s / coolant.rho_kg_m3 * 60000.0


def _velocity(flow_lpm: float, diameter_m: float) -> float:
    if diameter_m <= 0:
        raise ValueError("Pipe diameter must be positive")
    q_m3_s = flow_lpm / 60000.0
    area = math.pi * diameter_m**2 / 4.0
    return q_m3_s / area


def _friction_factor_swamee_jain(rho: float, mu: float, velocity: float, d: float, eps: float) -> float:
    if mu <= 0 or rho <= 0 or d <= 0:
        raise ValueError("Fluid properties and diameter must be positive")
    re = rho * velocity * d / mu
    if re <= 0:
        return 0.0
    if re < 2300:
        return 64.0 / re
    return 0.25 / (math.log10(eps / (3.7 * d) + 5.74 / (re**0.9)) ** 2)


def _pipe_dp_kpa(flow_lpm: float, rho: float, mu: float, length_m: float, d_m: float, eps_m: float) -> float:
    if length_m < 0 or eps_m < 0:
        raise ValueError("Length and roughness cannot be negative")
    v = _velocity(flow_lpm, d_m)
    f = _friction_factor_swamee_jain(rho, mu, v, d_m, eps_m)
    return f * (length_m / d_m) * (rho * v**2 / 2.0) / 1000.0


def _minor_dp_kpa(flow_lpm: float, rho: float, d_m: float, k: float) -> float:
    if k < 0:
        raise ValueError("Minor-loss K cannot be negative")
    v = _velocity(flow_lpm, d_m)
    return k * (rho * v**2 / 2.0) / 1000.0


def _max_liquid_racks_in_one_row(racks: pd.DataFrame, pod: str) -> int:
    """Dynamic row concurrency. Falls back to all liquid racks if no row column exists."""
    x = heat_loads(racks)
    x = x[(x["pod"].astype(str) == str(pod)) & x["liquid_cooled"]]
    if x.empty:
        return 0
    if "row" not in x.columns:
        return int(len(x))
    counts = x.groupby("row")["rack_id"].count()
    return int(counts.max()) if not counts.empty else 0


def hydraulic_candidate(
    pod_liquid_load_kw: float,
    liquid_racks: int,
    coolant: Coolant,
    delta_t_k: float,
    geom: HydraulicGeometry,
    max_liquid_per_row: int | None = None,
    water_ref_rack_flow_lpm: float | None = None,
    water_ref_rho: float = 992.2,
) -> dict:
    if liquid_racks < 0:
        raise ValueError("liquid_racks cannot be negative")
    rack_q_kw = 0.0 if liquid_racks == 0 else pod_liquid_load_kw / liquid_racks
    rack_flow = required_flow_lpm(rack_q_kw, coolant, delta_t_k) if liquid_racks else 0.0
    pod_flow = rack_flow * liquid_racks
    if max_liquid_per_row is None:
        max_liquid_per_row = liquid_racks
    row_flow = rack_flow * min(max(max_liquid_per_row, 0), liquid_racks)

    dp_common = _pipe_dp_kpa(
        pod_flow, coolant.rho_kg_m3, coolant.mu_pa_s,
        geom.common_length_m, geom.common_diameter_m, geom.roughness_m,
    )
    dp_row = _pipe_dp_kpa(
        row_flow, coolant.rho_kg_m3, coolant.mu_pa_s,
        geom.row_length_m, geom.row_diameter_m, geom.roughness_m,
    ) if row_flow > 0 else 0.0
    dp_branch = _pipe_dp_kpa(
        rack_flow, coolant.rho_kg_m3, coolant.mu_pa_s,
        geom.branch_length_m, geom.branch_diameter_m, geom.roughness_m,
    ) if rack_flow > 0 else 0.0
    dp_minor = (
        _minor_dp_kpa(pod_flow, coolant.rho_kg_m3, geom.common_diameter_m, geom.common_minor_k)
        + (_minor_dp_kpa(rack_flow, coolant.rho_kg_m3, geom.branch_diameter_m, geom.branch_minor_k) if rack_flow > 0 else 0.0)
    )
    network_dp = dp_common + dp_row + dp_branch + dp_minor

    if liquid_racks == 0:
        rack_dp = 0.0
    elif water_ref_rack_flow_lpm is None or water_ref_rack_flow_lpm <= 0:
        rack_dp = geom.rack_dp_reference_kpa
    else:
        rack_dp = geom.rack_dp_reference_kpa * (coolant.rho_kg_m3 / water_ref_rho) * (rack_flow / water_ref_rack_flow_lpm) ** 2

    total_dp = network_dp + rack_dp
    eta = max(geom.pump_efficiency * geom.motor_efficiency, 1e-6)
    pump_kw = (total_dp * 1000.0) * (pod_flow / 60000.0) / 1000.0 / eta

    return {
        "rack_avg_heat_kw": rack_q_kw,
        "rack_flow_lpm": rack_flow,
        "pod_flow_lpm": pod_flow,
        "common_velocity_m_s": _velocity(pod_flow, geom.common_diameter_m),
        "row_velocity_m_s": _velocity(row_flow, geom.row_diameter_m) if row_flow > 0 else 0.0,
        "branch_velocity_m_s": _velocity(rack_flow, geom.branch_diameter_m) if rack_flow > 0 else 0.0,
        "network_dp_kpa": network_dp,
        "rack_dp_kpa": rack_dp,
        "total_dp_kpa": total_dp,
        "pump_kw": pump_kw,
    }


def evaluate_coolants(
    racks: pd.DataFrame,
    coolants: list[Coolant],
    delta_t_k: float,
    geom: HydraulicGeometry,
) -> pd.DataFrame:
    pods = pod_summary(racks)
    water_ref = Coolant("Water reference", 992.2, 4.179, 0.000653)
    x = heat_loads(racks)
    liquid = x[x["liquid_cooled"]]
    ref_q = float(liquid["liquid_load_kw"].mean()) if len(liquid) else 0.0
    water_ref_flow = required_flow_lpm(ref_q, water_ref, delta_t_k) if ref_q > 0 else 0.0

    records = []
    for fluid in coolants:
        for _, row in pods.iterrows():
            pod_name = str(row["pod"])
            max_per_row = _max_liquid_racks_in_one_row(racks, pod_name)
            result = hydraulic_candidate(
                pod_liquid_load_kw=float(row["liquid_load_kw"]),
                liquid_racks=int(row["liquid_racks"]),
                coolant=fluid,
                delta_t_k=delta_t_k,
                geom=geom,
                max_liquid_per_row=max_per_row,
                water_ref_rack_flow_lpm=water_ref_flow,
                water_ref_rho=water_ref.rho_kg_m3,
            )
            records.append({
                "coolant": fluid.name,
                "pod": pod_name,
                "rack_positions": int(row["rack_positions"]),
                "liquid_racks": int(row["liquid_racks"]),
                "liquid_load_kw": float(row["liquid_load_kw"]),
                "max_liquid_racks_per_row": max_per_row,
                **result,
            })
    return pd.DataFrame(records)


def candidate_score_table(hydraulics: pd.DataFrame, cdu_capacity_mw: float) -> pd.DataFrame:
    """Transparent PoC sensitivity ranking; not equipment/coolant certification."""
    if cdu_capacity_mw <= 0:
        raise ValueError("CDU capacity must be positive")
    agg = hydraulics.groupby("coolant", as_index=False).agg(
        total_pump_kw=("pump_kw", "sum"),
        worst_dp_kpa=("total_dp_kpa", "max"),
        worst_velocity_m_s=("branch_velocity_m_s", "max"),
        max_pod_load_kw=("liquid_load_kw", "max"),
    )
    agg["cdu_loading_pct"] = agg["max_pod_load_kw"] / (cdu_capacity_mw * 1000.0) * 100.0
    pmin, pmax = agg["total_pump_kw"].min(), agg["total_pump_kw"].max()
    dmin, dmax = agg["worst_dp_kpa"].min(), agg["worst_dp_kpa"].max()
    agg["energy_index"] = 0.0 if pmax == pmin else (agg["total_pump_kw"] - pmin) / (pmax - pmin)
    agg["hydraulic_index"] = 0.0 if dmax == dmin else (agg["worst_dp_kpa"] - dmin) / (dmax - dmin)
    agg["balanced_score"] = 100.0 - 50.0 * agg["energy_index"] - 50.0 * agg["hydraulic_index"]
    agg = agg.sort_values(["balanced_score", "total_pump_kw"], ascending=[False, True]).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    return agg


def recommended_duty_cdus(total_liquid_kw: float, cdu_capacity_mw: float) -> int:
    if cdu_capacity_mw <= 0:
        raise ValueError("CDU capacity must be positive")
    return int(math.ceil(max(total_liquid_kw, 0.0) / (cdu_capacity_mw * 1000.0)))
