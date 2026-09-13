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
    loop_type: str = "Direct Return"
    balancing_margin_kpa: float = 15.0

@dataclass
class RackPressureCurve:
    """
    OEM rack pressure-flow curve.

    Model:
        deltaP_kPa = a * Q_lpm^2 + b * Q_lpm

    The curve is fitted only from engineer-verified
    OEM operating points.
    """

    a: float
    b: float
    q_min_lpm: float
    q_max_lpm: float
    point_count: int


def fit_rack_dp_curve(
    points: list[tuple[float, float]],
) -> RackPressureCurve:
    """
    Fit an OEM rack pressure-flow curve:

        deltaP = a*Q^2 + b*Q

    At least two verified operating points are required.
    The model is constrained to pass through the origin,
    which is physically appropriate for zero-flow loss.
    """

    if len(points) < 2:
        raise ValueError(
            "At least two rack flow / pressure-drop points "
            "are required for curve fitting."
        )

    cleaned = []

    for flow_lpm, dp_kpa in points:
        flow = float(flow_lpm)
        dp = float(dp_kpa)

        if flow <= 0:
            raise ValueError(
                "Rack flow points must be greater than 0 L/min."
            )

        if dp <= 0:
            raise ValueError(
                "Rack pressure-drop points must be greater than 0 kPa."
            )

        cleaned.append(
            (flow, dp)
        )

    # Prevent duplicate flow points
    flows = [
        item[0]
        for item in cleaned
    ]

    if len(set(flows)) != len(flows):
        raise ValueError(
            "Duplicate rack flow points are not allowed."
        )

    q = np.array(
        flows,
        dtype=float,
    )

    dp = np.array(
        [
            item[1]
            for item in cleaned
        ],
        dtype=float,
    )

    # deltaP = a*Q^2 + b*Q
    x = np.column_stack(
        [
            q**2,
            q,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        x,
        dp,
        rcond=None,
    )

    a = float(
        coefficients[0]
    )

    b = float(
        coefficients[1]
    )

    q_min = float(
        q.min()
    )

    q_max = float(
        q.max()
    )

    # Check that the fitted relationship remains
    # physically increasing throughout the
    # verified OEM data range.
    derivative_min = min(
        2.0 * a * q_min + b,
        2.0 * a * q_max + b,
    )

    if derivative_min <= 0:
        raise ValueError(
            "The fitted rack pressure-flow curve is not "
            "monotonically increasing over the OEM data range. "
            "Review the source operating points."
        )

    return RackPressureCurve(
        a=a,
        b=b,
        q_min_lpm=q_min,
        q_max_lpm=q_max,
        point_count=len(cleaned),
    )


def rack_dp_from_curve(
    flow_lpm: float,
    curve: RackPressureCurve,
    allow_extrapolation: bool = False,
) -> float:
    """
    Calculate rack pressure drop from an OEM-fitted curve.

    By default, extrapolation outside the verified OEM
    flow range is blocked.
    """

    flow = float(
        flow_lpm
    )

    if flow <= 0:
        return 0.0

    if not allow_extrapolation:
        if (
            flow < curve.q_min_lpm
            or flow > curve.q_max_lpm
        ):
            raise ValueError(
                f"Calculated rack flow {flow:.1f} L/min is outside "
                f"the verified OEM curve range "
                f"{curve.q_min_lpm:.1f}–{curve.q_max_lpm:.1f} L/min."
            )

    dp = (
        curve.a * flow**2
        + curve.b * flow
    )

    if dp < 0:
        raise ValueError(
            "Calculated rack pressure drop became negative. "
            "Review the OEM pressure-flow curve."
        )

    return float(dp)

# =========================================================
# Coolant Temperature / Property Helpers
# =========================================================
def bulk_temperature_c(
    supply_temp_c: float,
    return_temp_c: float,
) -> float:
    """
    Calculate the mean bulk coolant temperature
    for preliminary temperature-dependent property evaluation.

    T_bulk = (T_supply + T_return) / 2
    """

    supply = float(
        supply_temp_c
    )

    return_temp = float(
        return_temp_c
    )

    if return_temp <= supply:
        raise ValueError(
            "Return temperature must be greater than "
            "supply temperature."
        )

    return (
        supply
        + return_temp
    ) / 2.0


def interpolate_coolant_properties(
    property_points: list[dict],
    target_temp_c: float,
) -> dict:
    """
    Interpolate coolant properties at a target bulk temperature.

    Expected point format:
    {
        "temperature_c": 30.0,
        "density_kg_m3": 1025.0,
        "cp_kj_kgk": 3.88,
        "viscosity_mpas": 1.45,
    }

    Interpolation:
    - Density: linear
    - Specific heat: linear
    - Dynamic viscosity: logarithmic

    Extrapolation outside the verified temperature
    range is not permitted.
    """

    if len(property_points) < 2:
        raise ValueError(
            "At least two coolant property-temperature points "
            "are required for interpolation."
        )

    cleaned_points = []

    for point in property_points:
        temp = point.get(
            "temperature_c"
        )

        rho = point.get(
            "density_kg_m3"
        )

        cp = point.get(
            "cp_kj_kgk"
        )

        mu = point.get(
            "viscosity_mpas"
        )

        if (
            temp is None
            or rho is None
            or cp is None
            or mu is None
        ):
            continue

        temp = float(temp)
        rho = float(rho)
        cp = float(cp)
        mu = float(mu)

        if rho <= 0:
            raise ValueError(
                "Coolant density must be greater than 0."
            )

        if cp <= 0:
            raise ValueError(
                "Coolant specific heat must be greater than 0."
            )

        if mu <= 0:
            raise ValueError(
                "Coolant viscosity must be greater than 0."
            )

        cleaned_points.append(
            {
                "temperature_c": temp,
                "density_kg_m3": rho,
                "cp_kj_kgk": cp,
                "viscosity_mpas": mu,
            }
        )

    if len(cleaned_points) < 2:
        raise ValueError(
            "At least two complete coolant property points "
            "are required for interpolation."
        )

    cleaned_points = sorted(
        cleaned_points,
        key=lambda x: x[
            "temperature_c"
        ],
    )

    temperatures = [
        point["temperature_c"]
        for point in cleaned_points
    ]

    if (
        len(set(temperatures))
        != len(temperatures)
    ):
        raise ValueError(
            "Duplicate coolant property temperatures "
            "are not allowed."
        )

    target = float(
        target_temp_c
    )

    temp_min = temperatures[0]
    temp_max = temperatures[-1]

    if (
        target < temp_min
        or target > temp_max
    ):
        raise ValueError(
            f"Bulk temperature {target:.1f}°C is outside "
            f"the verified coolant property range "
            f"{temp_min:.1f}–{temp_max:.1f}°C."
        )

    # Exact verified table point
    for point in cleaned_points:
        if math.isclose(
            target,
            point["temperature_c"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return {
                "temperature_c": target,
                "density_kg_m3": point[
                    "density_kg_m3"
                ],
                "cp_kj_kgk": point[
                    "cp_kj_kgk"
                ],
                "viscosity_mpas": point[
                    "viscosity_mpas"
                ],
                "interpolation_basis": (
                    "Exact verified property-table point"
                ),
            }

    lower = None
    upper = None

    for index in range(
        len(cleaned_points) - 1
    ):
        point_1 = cleaned_points[
            index
        ]

        point_2 = cleaned_points[
            index + 1
        ]

        if (
            point_1["temperature_c"]
            <= target
            <= point_2["temperature_c"]
        ):
            lower = point_1
            upper = point_2
            break

    if (
        lower is None
        or upper is None
    ):
        raise ValueError(
            "Could not locate interpolation interval "
            "for coolant properties."
        )

    t1 = lower[
        "temperature_c"
    ]

    t2 = upper[
        "temperature_c"
    ]

    fraction = (
        target - t1
    ) / (
        t2 - t1
    )

    rho = (
        lower["density_kg_m3"]
        + fraction
        * (
            upper["density_kg_m3"]
            - lower["density_kg_m3"]
        )
    )

    cp = (
        lower["cp_kj_kgk"]
        + fraction
        * (
            upper["cp_kj_kgk"]
            - lower["cp_kj_kgk"]
        )
    )

    log_mu_1 = math.log(
        lower["viscosity_mpas"]
    )

    log_mu_2 = math.log(
        upper["viscosity_mpas"]
    )

    viscosity_mpas = math.exp(
        log_mu_1
        + fraction
        * (
            log_mu_2
            - log_mu_1
        )
    )

    return {
        "temperature_c": target,
        "density_kg_m3": float(
            rho
        ),
        "cp_kj_kgk": float(
            cp
        ),
        "viscosity_mpas": float(
            viscosity_mpas
        ),
        "interpolation_basis": (
            f"Interpolated between "
            f"{t1:.1f}°C and {t2:.1f}°C"
        ),
    }
    
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

def build_cfd_boundary_conditions(
    racks: pd.DataFrame,
    coolant: Coolant,
    supply_temp_c: float,
    return_temp_c: float,
    rack_pitch_m: float | None = None,
    row_pitch_m: float | None = None,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
) -> pd.DataFrame:
    """
    Build rack-level boundary-condition data for
    downstream CFD / thermal analysis handoff.

    Optional x/y coordinates are generated only when
    rack_pitch_m and row_pitch_m are both supplied.

    The generated coordinates are preliminary layout
    coordinates derived from rack row/column indices.
    This function does not perform CFD.
    """

    supply_temp = float(
        supply_temp_c
    )

    return_temp = float(
        return_temp_c
    )

    if return_temp <= supply_temp:
        raise ValueError(
            "Return temperature must be greater than "
            "supply temperature."
        )

    delta_t = (
        return_temp
        - supply_temp
    )

    rack_data = heat_loads(
        racks
    ).copy()

    # =========================================
    # Rack-level thermal / liquid boundaries
    # =========================================
    rack_data[
        "required_liquid_flow_lpm"
    ] = rack_data[
        "liquid_load_kw"
    ].apply(
        lambda heat_kw: required_flow_lpm(
            float(heat_kw),
            coolant,
            delta_t,
        )
        if float(heat_kw) > 0
        else 0.0
    )

    rack_data[
        "supply_temp_c"
    ] = supply_temp

    rack_data[
        "return_temp_c"
    ] = return_temp

    rack_data[
        "delta_t_k"
    ] = delta_t

    rack_data[
        "coolant"
    ] = coolant.name

    rack_data[
        "coolant_density_kg_m3"
    ] = float(
        coolant.rho_kg_m3
    )

    rack_data[
        "coolant_cp_kj_kgk"
    ] = float(
        coolant.cp_kj_kgk
    )

    rack_data[
        "coolant_viscosity_mpas"
    ] = float(
        coolant.mu_pa_s
        * 1000.0
    )

    # =========================================
    # Optional layout coordinate generation
    # =========================================
    coordinate_columns = []

    pitch_requested = (
        rack_pitch_m is not None
        or row_pitch_m is not None
    )

    if pitch_requested:
        if (
            rack_pitch_m is None
            or row_pitch_m is None
        ):
            raise ValueError(
                "Both rack_pitch_m and row_pitch_m "
                "must be supplied to generate x/y coordinates."
            )

        rack_pitch = float(
            rack_pitch_m
        )

        row_pitch = float(
            row_pitch_m
        )

        if rack_pitch <= 0:
            raise ValueError(
                "Rack pitch must be greater than 0 m."
            )

        if row_pitch <= 0:
            raise ValueError(
                "Row pitch must be greater than 0 m."
            )

        if (
            "row" not in rack_data.columns
            or "col" not in rack_data.columns
        ):
            raise ValueError(
                "Rack row/col data are required "
                "to generate x/y coordinates."
            )

        def axis_sort_key(
            value,
        ):
            try:
                return (
                    0,
                    float(value),
                )
            except (
                TypeError,
                ValueError,
            ):
                return (
                    1,
                    str(value),
                )

        unique_rows = sorted(
            rack_data[
                "row"
            ].drop_duplicates().tolist(),
            key=axis_sort_key,
        )

        unique_cols = sorted(
            rack_data[
                "col"
            ].drop_duplicates().tolist(),
            key=axis_sort_key,
        )

        row_index_map = {
            value: index
            for index, value
            in enumerate(
                unique_rows
            )
        }

        col_index_map = {
            value: index
            for index, value
            in enumerate(
                unique_cols
            )
        }

        rack_data[
            "x_m"
        ] = rack_data[
            "col"
        ].map(
            col_index_map
        ).astype(
            float
        ) * rack_pitch + float(
            origin_x_m
        )

        rack_data[
            "y_m"
        ] = rack_data[
            "row"
        ].map(
            row_index_map
        ).astype(
            float
        ) * row_pitch + float(
            origin_y_m
        )

        coordinate_columns = [
            "x_m",
            "y_m",
        ]

    # =========================================
    # Export schema
    # =========================================
    output_columns = [
        "rack_id",
        "pod",
    ]

    if "row" in rack_data.columns:
        output_columns.append(
            "row"
        )

    if "col" in rack_data.columns:
        output_columns.append(
            "col"
        )

    output_columns.extend(
        coordinate_columns
    )

    output_columns.extend(
        [
            "it_power_kw",
            "hcr",
            "liquid_load_kw",
            "residual_air_kw",
            "required_liquid_flow_lpm",
            "supply_temp_c",
            "return_temp_c",
            "delta_t_k",
            "coolant",
            "coolant_density_kg_m3",
            "coolant_cp_kj_kgk",
            "coolant_viscosity_mpas",
        ]
    )

    return rack_data[
        output_columns
    ].copy()
    
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
    rack_dp_curve: RackPressureCurve | None = None,
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
    
    # =========================================
    # Preliminary Rack-Path Imbalance Model
    # =========================================
    active_racks_in_row = min(
        max(
            int(max_liquid_per_row or 0),
            0,
        ),
        liquid_racks,
    )

    if active_racks_in_row <= 0:
        active_racks_in_row = (
            1 if liquid_racks > 0 else 0
        )

    loop_type = str(
        geom.loop_type
    ).strip()

    # row_length_m is treated as the
    # worst-case combined equivalent header path
    # for this preliminary model.
    if active_racks_in_row == 0:
        near_row_length_m = 0.0
        far_row_length_m = 0.0

    elif loop_type == "Direct Return":
        # In a direct-return arrangement, the rack
        # nearest the row connection has the shortest
        # equivalent header path and the farthest rack
        # has the longest.
        near_row_length_m = (
            geom.row_length_m
            / active_racks_in_row
        )

        far_row_length_m = (
            geom.row_length_m
        )

    elif loop_type == "Reverse Return / Tichelmann":
        # Preliminary reverse-return assumption:
        # supply + return path lengths are approximately
        # hydraulically equalized across rack positions.
        near_row_length_m = (
            geom.row_length_m
        )

        far_row_length_m = (
            geom.row_length_m
        )

    else:
        raise ValueError(
            "Unsupported loop type. Use "
            "'Direct Return' or "
            "'Reverse Return / Tichelmann'."
        )

    near_row_dp = (
        _pipe_dp_kpa(
            row_flow,
            coolant.rho_kg_m3,
            coolant.mu_pa_s,
            near_row_length_m,
            geom.row_diameter_m,
            geom.roughness_m,
        )
        if row_flow > 0
        else 0.0
    )

    far_row_dp = (
        _pipe_dp_kpa(
            row_flow,
            coolant.rho_kg_m3,
            coolant.mu_pa_s,
            far_row_length_m,
            geom.row_diameter_m,
            geom.roughness_m,
        )
        if row_flow > 0
        else 0.0
    )

    # Common and rack-branch losses are shared
    # in this preliminary path comparison.
    non_row_network_dp = (
        dp_common
        + dp_branch
        + dp_minor
    )

    near_network_dp = (
        non_row_network_dp
        + near_row_dp
    )

    far_network_dp = (
        non_row_network_dp
        + far_row_dp
    )

    path_imbalance_kpa = abs(
        far_network_dp
        - near_network_dp
    )

    network_dp = max(
        near_network_dp,
        far_network_dp,
    )

    if (
        math.isclose(
            near_network_dp,
            far_network_dp,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        worst_case_rack = (
            "Hydraulically equivalent rack path"
        )

    elif far_network_dp > near_network_dp:
        worst_case_rack = (
            "Far-end rack"
        )

    else:
        worst_case_rack = (
            "Near-end rack"
        )

    balancing_margin_kpa = max(
        float(
            geom.balancing_margin_kpa
        ),
        0.0,
    )

    # =========================================
    # Rack internal pressure drop
    # =========================================
    if liquid_racks == 0:
        rack_dp = 0.0
        rack_dp_basis = "No liquid flow"

    elif rack_dp_curve is not None:
        rack_dp = rack_dp_from_curve(
            rack_flow,
            rack_dp_curve,
            allow_extrapolation=False,
        )

        rack_dp_basis = (
            "OEM multi-point pressure-flow curve"
        )

    elif (
        water_ref_rack_flow_lpm is None
        or water_ref_rack_flow_lpm <= 0
    ):
        rack_dp = (
            geom.rack_dp_reference_kpa
        )

        rack_dp_basis = (
            "Synthetic fixed placeholder"
        )

    else:
        rack_dp = (
            geom.rack_dp_reference_kpa
            * (
                coolant.rho_kg_m3
                / water_ref_rho
            )
            * (
                rack_flow
                / water_ref_rack_flow_lpm
            ) ** 2
        )

        rack_dp_basis = (
            "Synthetic single-point square-law scaling"
        )

    total_dp = (
        network_dp
        + rack_dp
        + balancing_margin_kpa
    )

    eta = max(
        geom.pump_efficiency
        * geom.motor_efficiency,
        1e-6,
    )
    pump_kw = (total_dp * 1000.0) * (pod_flow / 60000.0) / 1000.0 / eta

    return {
        "rack_avg_heat_kw": rack_q_kw,
        "rack_flow_lpm": rack_flow,
        "pod_flow_lpm": pod_flow,
        "common_velocity_m_s": _velocity(pod_flow, geom.common_diameter_m),
        "row_velocity_m_s": _velocity(row_flow, geom.row_diameter_m) if row_flow > 0 else 0.0,
        "branch_velocity_m_s": _velocity(rack_flow, geom.branch_diameter_m) if rack_flow > 0 else 0.0,
        "loop_type": loop_type,

        "near_row_length_m": (
            near_row_length_m
        ),

        "far_row_length_m": (
            far_row_length_m
        ),

        "near_network_dp_kpa": (
            near_network_dp
        ),

        "far_network_dp_kpa": (
            far_network_dp
        ),

        "path_imbalance_kpa": (
            path_imbalance_kpa
        ),

        "balancing_margin_kpa": (
            balancing_margin_kpa
        ),

        "worst_case_rack": (
            worst_case_rack
        ),
        "network_dp_kpa": network_dp,
        "rack_dp_kpa": rack_dp,
        "rack_dp_basis": rack_dp_basis,
        "total_dp_kpa": total_dp,
        "pump_kw": pump_kw,
    }


def evaluate_coolants(
    racks: pd.DataFrame,
    coolants: list[Coolant],
    delta_t_k: float,
    geom: HydraulicGeometry,
    rack_dp_curve: RackPressureCurve | None = None,
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
                rack_dp_curve=rack_dp_curve,
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
