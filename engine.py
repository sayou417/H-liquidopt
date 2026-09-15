from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import least_squares
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
class HydraulicNetworkLayout:
    """
    Physical layout inputs for detailed rack-level
    hydraulic network analysis.

    Coordinates are project-local coordinates in metres.

    rack_pitch_m:
        Centre-to-centre spacing between adjacent rack columns.

    row_pitch_m:
        Centre-to-centre spacing between rack rows.

    pod_pitch_m:
        Representative centre-to-centre spacing between Pods.
        Used when a Central CDU topology creates a shared
        main-header relationship between multiple Pods.

    origin_x_m / origin_y_m:
        Coordinate of the first rack position.

    cdu_y_m:
        Y-coordinate of the CDU / common-header connection.

    supply_header_x_m:
        X-coordinate where the supply header enters each row.

    return_header_x_m:
        X-coordinate where the return header leaves each row.

    topology_mode:
        Approved Phase 2 CDU topology.
        Supported values:
        - pod_dedicated
        - central
        - in_row
    """

    rack_pitch_m: float = 0.8
    row_pitch_m: float = 4.0
    pod_pitch_m: float = 12.0

    origin_x_m: float = 0.0
    origin_y_m: float = 0.0

    cdu_y_m: float = -2.0

    supply_header_x_m: float = 0.0
    return_header_x_m: float = 0.0

    topology_mode: str = "pod_dedicated"

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

def rack_flow_requirements(
    racks: pd.DataFrame,
    coolant: Coolant,
    delta_t_k: float,
) -> pd.DataFrame:
    """
    Build rack-level hydraulic demand data.

    Unlike the preliminary pod-level model, this function
    preserves each rack's individual heat load and calculates
    its own thermal required liquid flow.

    This is the input layer for the detailed hydraulic
    network solver.
    """

    if delta_t_k <= 0:
        raise ValueError(
            "delta_t_k must be greater than 0."
        )

    rack_data = heat_loads(
        racks
    ).copy()

    rack_data = rack_data[
        rack_data["liquid_cooled"]
    ].copy()

    if rack_data.empty:
        return pd.DataFrame(
            columns=[
                "rack_id",
                "pod",
                "row",
                "col",
                "liquid_load_kw",
                "required_flow_lpm",
            ]
        )

    rack_data[
        "required_flow_lpm"
    ] = rack_data[
        "liquid_load_kw"
    ].apply(
        lambda heat_kw: required_flow_lpm(
            float(heat_kw),
            coolant,
            delta_t_k,
        )
    )

    # -----------------------------------------
    # Preserve layout information
    # -----------------------------------------
    if "row" not in rack_data.columns:
        raise ValueError(
            "Detailed hydraulic network analysis "
            "requires a 'row' column."
        )

    if "col" not in rack_data.columns:
        raise ValueError(
            "Detailed hydraulic network analysis "
            "requires a 'col' column."
        )

    if rack_data[
        [
            "row",
            "col",
        ]
    ].isna().any().any():
        raise ValueError(
            "Detailed hydraulic network analysis "
            "requires row/col for every liquid-cooled rack."
        )

    # -----------------------------------------
    # Create stable positional indices
    # without assuming row/col are physical metres
    # -----------------------------------------
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

    # =========================================
    # Pod-local row / column indices
    # =========================================
    rack_data[
        "pod_index"
    ] = 0

    rack_data[
        "row_index"
    ] = 0

    rack_data[
        "col_index"
    ] = 0

    unique_pods = sorted(
        rack_data[
            "pod"
        ].astype(
            str
        ).drop_duplicates().tolist(),
        key=axis_sort_key,
    )

    for pod_index, pod_value in enumerate(
        unique_pods
    ):
        pod_mask = (
            rack_data[
                "pod"
            ].astype(
                str
            )
            == str(
                pod_value
            )
        )

        rack_data.loc[
            pod_mask,
            "pod_index",
        ] = int(
            pod_index
        )

        pod_rows = sorted(
            rack_data.loc[
                pod_mask,
                "row",
            ].drop_duplicates().tolist(),
            key=axis_sort_key,
        )

        row_index_map = {
            value: index
            for index, value
            in enumerate(
                pod_rows
            )
        }

        rack_data.loc[
            pod_mask,
            "row_index",
        ] = (
            rack_data.loc[
                pod_mask,
                "row",
            ]
            .map(
                row_index_map
            )
            .astype(
                int
            )
        )

        for row_value in pod_rows:
            row_mask = (
                pod_mask
                & (
                    rack_data[
                        "row"
                    ]
                    == row_value
                )
            )

            row_columns = sorted(
                rack_data.loc[
                    row_mask,
                    "col",
                ].drop_duplicates().tolist(),
                key=axis_sort_key,
            )

            col_index_map = {
                value: index
                for index, value
                in enumerate(
                    row_columns
                )
            }

            rack_data.loc[
                row_mask,
                "col_index",
            ] = (
                rack_data.loc[
                    row_mask,
                    "col",
                ]
                .map(
                    col_index_map
                )
                .astype(
                    int
                )
            )

    rack_data[
        [
            "pod_index",
            "row_index",
            "col_index",
        ]
    ] = rack_data[
        [
            "pod_index",
            "row_index",
            "col_index",
        ]
    ].astype(
        int
    )

    output_columns = [
        "rack_id",
        "pod",
        "pod_index",
        "row",
        "col",
        "row_index",
        "col_index",
        "it_power_kw",
        "hcr",
        "liquid_load_kw",
        "required_flow_lpm",
    ]

    return (
        rack_data[
            output_columns
        ]
        .sort_values(
            [
                "pod",
                "row_index",
                "col_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )

def build_rack_network_paths(
    racks: pd.DataFrame,
    coolant: Coolant,
    delta_t_k: float,
    layout: HydraulicNetworkLayout,
) -> pd.DataFrame:
    """
    Convert rack row/column locations into physical
    coordinates and supply/return hydraulic path lengths.

    This function builds geometry only.
    It does not solve rack flow distribution.
    """

    if layout.rack_pitch_m <= 0:
        raise ValueError(
            "rack_pitch_m must be greater than 0."
        )

    if layout.row_pitch_m <= 0:
        raise ValueError(
            "row_pitch_m must be greater than 0."
        )

    rack_data = rack_flow_requirements(
        racks,
        coolant,
        delta_t_k,
    ).copy()

    if rack_data.empty:
        return rack_data

    # =========================================
    # Physical rack coordinates
    # =========================================
    rack_data[
        "x_m"
    ] = (
        float(layout.origin_x_m)
        + rack_data[
            "col_index"
        ].astype(float)
        * float(layout.rack_pitch_m)
    )

    # =========================================
    # Pod / Row physical Y position
    # =========================================
    topology_mode = str(
        layout.topology_mode
    ).strip()

    if topology_mode not in {
        "pod_dedicated",
        "central",
        "in_row",
    }:
        raise ValueError(
            "Unsupported topology_mode. Use "
            "'pod_dedicated', 'central', or 'in_row'."
        )

    if topology_mode == "central":
        if layout.pod_pitch_m <= 0:
            raise ValueError(
                "pod_pitch_m must be greater than 0 "
                "for Central CDU topology."
            )

        pod_y_offset = (
            rack_data[
                "pod_index"
            ].astype(float)
            * float(
                layout.pod_pitch_m
            )
        )

    else:
        # Pod-dedicated and In-row systems are
        # treated as local hydraulic subsystems.
        pod_y_offset = 0.0

    rack_data[
        "y_m"
    ] = (
        float(
            layout.origin_y_m
        )
        + pod_y_offset
        + rack_data[
            "row_index"
        ].astype(float)
        * float(
            layout.row_pitch_m
        )
    )

    # =========================================
    # Common-header path from CDU to row
    # =========================================
    rack_data[
        "supply_common_length_m"
    ] = (
        rack_data["y_m"]
        - float(layout.cdu_y_m)
    ).abs()

    rack_data[
        "return_common_length_m"
    ] = (
        rack_data["y_m"]
        - float(layout.cdu_y_m)
    ).abs()

    # =========================================
    # Row-header path
    # =========================================
    rack_data[
        "supply_row_length_m"
    ] = (
        rack_data["x_m"]
        - float(
            layout.supply_header_x_m
        )
    ).abs()

    rack_data[
        "return_row_length_m"
    ] = (
        rack_data["x_m"]
        - float(
            layout.return_header_x_m
        )
    ).abs()

    # Combined physical header path
    rack_data[
        "total_header_path_m"
    ] = (
        rack_data[
            "supply_common_length_m"
        ]
        + rack_data[
            "return_common_length_m"
        ]
        + rack_data[
            "supply_row_length_m"
        ]
        + rack_data[
            "return_row_length_m"
        ]
    )

    output_columns = [
        "rack_id",
        "pod",
        "pod_index",
        "row",
        "col",
        "row_index",
        "col_index",
        "x_m",
        "y_m",
        "it_power_kw",
        "hcr",
        "liquid_load_kw",
        "required_flow_lpm",
        "supply_common_length_m",
        "return_common_length_m",
        "supply_row_length_m",
        "return_row_length_m",
        "total_header_path_m",
    ]

    return (
        rack_data[
            output_columns
        ]
        .sort_values(
            [
                "pod",
                "row_index",
                "col_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )

def build_hydraulic_network_segments(
    racks: pd.DataFrame,
    coolant: Coolant,
    delta_t_k: float,
    geom: HydraulicGeometry,
    layout: HydraulicNetworkLayout,
) -> pd.DataFrame:
    """
    Build a topology-aware hydraulic network segment table.

    Supported Phase 2 topology modes:

    pod_dedicated
        Each Pod is treated as an independent hydraulic subsystem.

    central
        All Pods share one common supply / return main.
        Pod spacing therefore affects common-main length and losses.

    in_row
        Each Pod-Row group is treated as a local hydraulic subsystem.
        Common-main losses are omitted in this preliminary model because
        the CDU/feed is assumed to be located locally near the Row.

    This function builds network geometry and a design-flow reference
    state. Actual rack-flow distribution is solved separately.
    """

    rack_paths = build_rack_network_paths(
        racks,
        coolant,
        delta_t_k,
        layout,
    )

    if rack_paths.empty:
        return pd.DataFrame()

    topology_mode = str(
        layout.topology_mode
    ).strip()

    if topology_mode not in {
        "pod_dedicated",
        "central",
        "in_row",
    }:
        raise ValueError(
            "Unsupported topology_mode. Use "
            "'pod_dedicated', 'central', or 'in_row'."
        )

    records = []

    # =========================================
    # Helper · Determine whether a rack path
    # uses a one-dimensional pipe interval
    # =========================================
    def path_uses_interval(
        connection_position: float,
        rack_position: float,
        segment_start: float,
        segment_end: float,
    ) -> bool:
        path_min = min(
            float(connection_position),
            float(rack_position),
        )

        path_max = max(
            float(connection_position),
            float(rack_position),
        )

        seg_min = min(
            float(segment_start),
            float(segment_end),
        )

        seg_max = max(
            float(segment_start),
            float(segment_end),
        )

        tolerance = 1e-9

        return (
            seg_min >= path_min - tolerance
            and seg_max <= path_max + tolerance
            and seg_max - seg_min > tolerance
        )

    # =========================================
    # Helper · Append one physical segment
    # =========================================
    def append_segment(
        segment_id: str,
        hydraulic_group: str,
        pod_label: str,
        segment_type: str,
        side: str,
        row_value,
        start_x_m: float,
        start_y_m: float,
        end_x_m: float,
        end_y_m: float,
        diameter_m: float,
        downstream_data: pd.DataFrame,
        minor_k: float = 0.0,
    ):
        length_m = math.sqrt(
            (
                float(end_x_m)
                - float(start_x_m)
            ) ** 2
            + (
                float(end_y_m)
                - float(start_y_m)
            ) ** 2
        )

        if length_m <= 1e-12:
            return

        downstream_rack_ids = tuple(
            downstream_data[
                "rack_id"
            ].astype(
                str
            ).tolist()
        )

        design_flow_lpm = float(
            downstream_data[
                "required_flow_lpm"
            ].sum()
        )

        if design_flow_lpm > 0:
            velocity_m_s = _velocity(
                design_flow_lpm,
                diameter_m,
            )

            pipe_dp_kpa = _pipe_dp_kpa(
                design_flow_lpm,
                coolant.rho_kg_m3,
                coolant.mu_pa_s,
                length_m,
                diameter_m,
                geom.roughness_m,
            )

            minor_dp_kpa = _minor_dp_kpa(
                design_flow_lpm,
                coolant.rho_kg_m3,
                diameter_m,
                minor_k,
            )

        else:
            velocity_m_s = 0.0
            pipe_dp_kpa = 0.0
            minor_dp_kpa = 0.0

        records.append(
            {
                "segment_id": segment_id,
                "hydraulic_group": hydraulic_group,
                "pod": pod_label,
                "row": row_value,
                "segment_type": segment_type,
                "side": side,
                "start_x_m": float(
                    start_x_m
                ),
                "start_y_m": float(
                    start_y_m
                ),
                "end_x_m": float(
                    end_x_m
                ),
                "end_y_m": float(
                    end_y_m
                ),
                "length_m": float(
                    length_m
                ),
                "diameter_m": float(
                    diameter_m
                ),
                "downstream_rack_count": int(
                    len(
                        downstream_rack_ids
                    )
                ),
                "downstream_rack_ids": (
                    downstream_rack_ids
                ),
                "design_flow_lpm": (
                    design_flow_lpm
                ),
                "design_velocity_m_s": (
                    velocity_m_s
                ),
                "design_pipe_dp_kpa": (
                    pipe_dp_kpa
                ),
                "minor_k": float(
                    minor_k
                ),
                "design_minor_dp_kpa": (
                    minor_dp_kpa
                ),
                "design_total_dp_kpa": (
                    pipe_dp_kpa
                    + minor_dp_kpa
                ),
            }
        )

    # =========================================
    # Build hydraulic subsystem groups
    # =========================================
    hydraulic_groups = []

    if topology_mode == "pod_dedicated":
        for pod_value in (
            rack_paths[
                "pod"
            ].astype(
                str
            ).drop_duplicates()
        ):
            group_data = rack_paths[
                rack_paths[
                    "pod"
                ].astype(
                    str
                )
                == str(
                    pod_value
                )
            ].copy()

            hydraulic_groups.append(
                (
                    f"POD::{pod_value}",
                    group_data,
                )
            )

    elif topology_mode == "central":
        hydraulic_groups.append(
            (
                "CENTRAL",
                rack_paths.copy(),
            )
        )

    else:
        unique_row_groups = (
            rack_paths[
                [
                    "pod",
                    "row",
                ]
            ]
            .drop_duplicates()
        )

        for _, group_row in (
            unique_row_groups.iterrows()
        ):
            pod_value = str(
                group_row[
                    "pod"
                ]
            )

            row_value = group_row[
                "row"
            ]

            group_data = rack_paths[
                (
                    rack_paths[
                        "pod"
                    ].astype(
                        str
                    )
                    == pod_value
                )
                & (
                    rack_paths[
                        "row"
                    ]
                    == row_value
                )
            ].copy()

            hydraulic_groups.append(
                (
                    (
                        f"ROW::{pod_value}"
                        f"::{row_value}"
                    ),
                    group_data,
                )
            )

    # =========================================
    # Build each hydraulic subsystem
    # =========================================
    for (
        hydraulic_group,
        group_data,
    ) in hydraulic_groups:

        if group_data.empty:
            continue

        # =====================================
        # 1 · COMMON SUPPLY / RETURN MAIN
        # =====================================
        #
        # In-row topology intentionally skips
        # the common main because the preliminary
        # model assumes local CDU/feed placement.
        #
        if topology_mode != "in_row":

            row_y_positions = sorted(
                group_data[
                    "y_m"
                ].astype(
                    float
                ).unique().tolist()
            )

            common_y_nodes = sorted(
                set(
                    row_y_positions
                    + [
                        float(
                            layout.cdu_y_m
                        )
                    ]
                )
            )

            for index in range(
                len(
                    common_y_nodes
                )
                - 1
            ):
                y1 = float(
                    common_y_nodes[
                        index
                    ]
                )

                y2 = float(
                    common_y_nodes[
                        index + 1
                    ]
                )

                # -----------------------------
                # Common supply main
                # -----------------------------
                supply_mask = group_data[
                    "y_m"
                ].apply(
                    lambda rack_y: path_uses_interval(
                        layout.cdu_y_m,
                        float(
                            rack_y
                        ),
                        y1,
                        y2,
                    )
                )

                supply_downstream = group_data[
                    supply_mask
                ]

                if not supply_downstream.empty:
                    append_segment(
                        segment_id=(
                            f"{hydraulic_group}"
                            f"_SUP_COMMON_{index + 1}"
                        ),
                        hydraulic_group=(
                            hydraulic_group
                        ),
                        pod_label=(
                            "MULTI"
                            if topology_mode
                            == "central"
                            else str(
                                group_data[
                                    "pod"
                                ].iloc[
                                    0
                                ]
                            )
                        ),
                        segment_type=(
                            "common_header"
                        ),
                        side="supply",
                        row_value=None,
                        start_x_m=float(
                            layout.supply_header_x_m
                        ),
                        start_y_m=y1,
                        end_x_m=float(
                            layout.supply_header_x_m
                        ),
                        end_y_m=y2,
                        diameter_m=float(
                            geom.common_diameter_m
                        ),
                        downstream_data=(
                            supply_downstream
                        ),
                    )

                # -----------------------------
                # Common return main
                # -----------------------------
                return_mask = group_data[
                    "y_m"
                ].apply(
                    lambda rack_y: path_uses_interval(
                        layout.cdu_y_m,
                        float(
                            rack_y
                        ),
                        y1,
                        y2,
                    )
                )

                return_downstream = group_data[
                    return_mask
                ]

                if not return_downstream.empty:
                    append_segment(
                        segment_id=(
                            f"{hydraulic_group}"
                            f"_RET_COMMON_{index + 1}"
                        ),
                        hydraulic_group=(
                            hydraulic_group
                        ),
                        pod_label=(
                            "MULTI"
                            if topology_mode
                            == "central"
                            else str(
                                group_data[
                                    "pod"
                                ].iloc[
                                    0
                                ]
                            )
                        ),
                        segment_type=(
                            "common_header"
                        ),
                        side="return",
                        row_value=None,
                        start_x_m=float(
                            layout.return_header_x_m
                        ),
                        start_y_m=y1,
                        end_x_m=float(
                            layout.return_header_x_m
                        ),
                        end_y_m=y2,
                        diameter_m=float(
                            geom.common_diameter_m
                        ),
                        downstream_data=(
                            return_downstream
                        ),
                    )

        # =====================================
        # 2 · ROW SUPPLY / RETURN HEADERS
        # =====================================
        #
        # Group by both Pod and Row so identical
        # Row labels in different Pods never merge.
        #
        row_groups = (
            group_data[
                [
                    "pod",
                    "row",
                ]
            ]
            .drop_duplicates()
        )

        for _, row_group in (
            row_groups.iterrows()
        ):
            actual_pod = str(
                row_group[
                    "pod"
                ]
            )

            row_value = row_group[
                "row"
            ]

            row_data = group_data[
                (
                    group_data[
                        "pod"
                    ].astype(
                        str
                    )
                    == actual_pod
                )
                & (
                    group_data[
                        "row"
                    ]
                    == row_value
                )
            ].copy()

            rack_x_positions = sorted(
                row_data[
                    "x_m"
                ].astype(
                    float
                ).unique().tolist()
            )

            row_y = float(
                row_data[
                    "y_m"
                ].iloc[
                    0
                ]
            )

            # -------------------------------
            # Supply header
            # -------------------------------
            supply_x_nodes = sorted(
                set(
                    rack_x_positions
                    + [
                        float(
                            layout.supply_header_x_m
                        )
                    ]
                )
            )

            for index in range(
                len(
                    supply_x_nodes
                )
                - 1
            ):
                x1 = float(
                    supply_x_nodes[
                        index
                    ]
                )

                x2 = float(
                    supply_x_nodes[
                        index + 1
                    ]
                )

                supply_mask = row_data[
                    "x_m"
                ].apply(
                    lambda rack_x: path_uses_interval(
                        layout.supply_header_x_m,
                        float(
                            rack_x
                        ),
                        x1,
                        x2,
                    )
                )

                downstream = row_data[
                    supply_mask
                ]

                if downstream.empty:
                    continue

                append_segment(
                    segment_id=(
                        f"{hydraulic_group}"
                        f"_POD_{actual_pod}"
                        f"_ROW_{row_value}"
                        f"_SUP_{index + 1}"
                    ),
                    hydraulic_group=(
                        hydraulic_group
                    ),
                    pod_label=(
                        actual_pod
                    ),
                    segment_type=(
                        "row_header"
                    ),
                    side="supply",
                    row_value=row_value,
                    start_x_m=x1,
                    start_y_m=row_y,
                    end_x_m=x2,
                    end_y_m=row_y,
                    diameter_m=float(
                        geom.row_diameter_m
                    ),
                    downstream_data=(
                        downstream
                    ),
                )

            # -------------------------------
            # Return header
            # -------------------------------
            return_x_nodes = sorted(
                set(
                    rack_x_positions
                    + [
                        float(
                            layout.return_header_x_m
                        )
                    ]
                )
            )

            for index in range(
                len(
                    return_x_nodes
                )
                - 1
            ):
                x1 = float(
                    return_x_nodes[
                        index
                    ]
                )

                x2 = float(
                    return_x_nodes[
                        index + 1
                    ]
                )

                return_mask = row_data[
                    "x_m"
                ].apply(
                    lambda rack_x: path_uses_interval(
                        layout.return_header_x_m,
                        float(
                            rack_x
                        ),
                        x1,
                        x2,
                    )
                )

                downstream = row_data[
                    return_mask
                ]

                if downstream.empty:
                    continue

                append_segment(
                    segment_id=(
                        f"{hydraulic_group}"
                        f"_POD_{actual_pod}"
                        f"_ROW_{row_value}"
                        f"_RET_{index + 1}"
                    ),
                    hydraulic_group=(
                        hydraulic_group
                    ),
                    pod_label=(
                        actual_pod
                    ),
                    segment_type=(
                        "row_header"
                    ),
                    side="return",
                    row_value=row_value,
                    start_x_m=x1,
                    start_y_m=row_y,
                    end_x_m=x2,
                    end_y_m=row_y,
                    diameter_m=float(
                        geom.row_diameter_m
                    ),
                    downstream_data=(
                        downstream
                    ),
                )

        # =====================================
        # 3 · RACK BRANCH EQUIVALENT SEGMENTS
        # =====================================
        for _, rack_row in (
            group_data.iterrows()
        ):
            rack_id = str(
                rack_row[
                    "rack_id"
                ]
            )

            actual_pod = str(
                rack_row[
                    "pod"
                ]
            )

            rack_flow = float(
                rack_row[
                    "required_flow_lpm"
                ]
            )

            branch_pipe_dp = (
                _pipe_dp_kpa(
                    rack_flow,
                    coolant.rho_kg_m3,
                    coolant.mu_pa_s,
                    geom.branch_length_m,
                    geom.branch_diameter_m,
                    geom.roughness_m,
                )
                if rack_flow > 0
                else 0.0
            )

            branch_minor_dp = (
                _minor_dp_kpa(
                    rack_flow,
                    coolant.rho_kg_m3,
                    geom.branch_diameter_m,
                    geom.branch_minor_k,
                )
                if rack_flow > 0
                else 0.0
            )

            records.append(
                {
                    "segment_id": (
                        f"{hydraulic_group}"
                        f"_RACK_{rack_id}"
                        "_BRANCH"
                    ),
                    "hydraulic_group": (
                        hydraulic_group
                    ),
                    "pod": actual_pod,
                    "row": rack_row[
                        "row"
                    ],
                    "segment_type": (
                        "rack_branch_equivalent"
                    ),
                    "side": "rack",
                    "start_x_m": float(
                        rack_row[
                            "x_m"
                        ]
                    ),
                    "start_y_m": float(
                        rack_row[
                            "y_m"
                        ]
                    ),
                    "end_x_m": float(
                        rack_row[
                            "x_m"
                        ]
                    ),
                    "end_y_m": float(
                        rack_row[
                            "y_m"
                        ]
                    ),
                    "length_m": float(
                        geom.branch_length_m
                    ),
                    "diameter_m": float(
                        geom.branch_diameter_m
                    ),
                    "downstream_rack_count": 1,
                    "downstream_rack_ids": (
                        rack_id,
                    ),
                    "design_flow_lpm": (
                        rack_flow
                    ),
                    "design_velocity_m_s": (
                        _velocity(
                            rack_flow,
                            geom.branch_diameter_m,
                        )
                        if rack_flow > 0
                        else 0.0
                    ),
                    "design_pipe_dp_kpa": (
                        branch_pipe_dp
                    ),
                    "minor_k": float(
                        geom.branch_minor_k
                    ),
                    "design_minor_dp_kpa": (
                        branch_minor_dp
                    ),
                    "design_total_dp_kpa": (
                        branch_pipe_dp
                        + branch_minor_dp
                    ),
                }
            )

    return pd.DataFrame(
        records
    )

def solve_rack_flow_distribution(
    racks: pd.DataFrame,
    coolant: Coolant,
    delta_t_k: float,
    geom: HydraulicGeometry,
    layout: HydraulicNetworkLayout,
    rack_dp_curve: RackPressureCurve | None = None,
) -> dict:
    """
    Solve rack-level hydraulic flow distribution.

    Distribution mode:
    - Total solved rack flow is constrained to the
      total thermal required flow.
    - All parallel rack paths within a Pod share
      the same hydraulic head.
    - Header-segment losses are recalculated from
      the actual flow passing through each segment.

    The result therefore shows how the available
    total flow distributes among racks.

    Balancing margin is NOT treated as a physical
    balancing-valve resistance in this solver.
    It is added only to the reported pump-head basis.
    """

    rack_requirements = rack_flow_requirements(
        racks,
        coolant,
        delta_t_k,
    )

    network_segments = (
        build_hydraulic_network_segments(
            racks,
            coolant,
            delta_t_k,
            geom,
            layout,
        )
    )

    if rack_requirements.empty:
        return {
            "rack_results": pd.DataFrame(),
            "segment_results": pd.DataFrame(),
            "pod_summary": pd.DataFrame(),
        }

    if network_segments.empty:
        raise ValueError(
            "Detailed hydraulic network contains no pipe segments."
        )

    # =========================================
    # Synthetic fallback reference
    # =========================================
    water_reference = Coolant(
        "Water reference",
        992.2,
        4.179,
        0.000653,
    )

    mean_liquid_load_kw = float(
        rack_requirements[
            "liquid_load_kw"
        ].mean()
    )

    water_ref_flow_lpm = (
        required_flow_lpm(
            mean_liquid_load_kw,
            water_reference,
            delta_t_k,
        )
        if mean_liquid_load_kw > 0
        else 0.0
    )

    rack_output_records = []
    segment_output_records = []
    pod_output_records = []

    # =========================================
    # Solve each topology-defined
    # hydraulic subsystem
    # =========================================
    topology_mode = str(
        layout.topology_mode
    ).strip()

    if (
        "hydraulic_group"
        not in network_segments.columns
    ):
        raise ValueError(
            "Hydraulic segment table does not contain "
            "hydraulic_group information."
        )

    hydraulic_groups = (
        network_segments[
            "hydraulic_group"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    for hydraulic_group in hydraulic_groups:

        pod_segments = network_segments[
            network_segments[
                "hydraulic_group"
            ].astype(str)
            == str(
                hydraulic_group
            )
        ].copy()

        # -------------------------------------
        # Determine which racks belong to
        # this hydraulic subsystem.
        #
        # Every rack has exactly one branch
        # equivalent segment.
        # -------------------------------------
        branch_segments = pod_segments[
            pod_segments[
                "segment_type"
            ]
            == "rack_branch_equivalent"
        ].copy()

        group_rack_ids = []

        for downstream_ids in branch_segments[
            "downstream_rack_ids"
        ]:
            if isinstance(
                downstream_ids,
                (
                    tuple,
                    list,
                    set,
                    np.ndarray,
                    pd.Series,
                ),
            ):
                group_rack_ids.extend(
                    [
                        str(rack_id)
                        for rack_id
                        in downstream_ids
                    ]
                )

            else:
                group_rack_ids.append(
                    str(
                        downstream_ids
                    )
                )

        group_rack_ids = list(
            dict.fromkeys(
                group_rack_ids
            )
        )

        if not group_rack_ids:
            continue

        pod_racks = rack_requirements[
            rack_requirements[
                "rack_id"
            ].astype(str).isin(
                group_rack_ids
            )
        ].copy()

        if pod_racks.empty:
            continue

        pod_racks[
            "rack_id"
        ] = pod_racks[
            "rack_id"
        ].astype(str)

        pod_racks = (
            pod_racks.sort_values(
                [
                    "pod",
                    "row_index",
                    "col_index",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        group_pods = (
            pod_racks[
                "pod"
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if len(group_pods) == 1:
            summary_pod_label = (
                group_pods[0]
            )

        else:
            summary_pod_label = (
                "MULTI"
            )

        rack_ids = (
            pod_racks[
                "rack_id"
            ].astype(str).tolist()
        )

        required_flows = (
            pod_racks[
                "required_flow_lpm"
            ].astype(float).to_numpy()
        )

        target_total_flow = float(
            required_flows.sum()
        )

        if target_total_flow <= 0:
            continue

        rack_index = {
            rack_id: index
            for index, rack_id
            in enumerate(rack_ids)
        }

        required_flow_map = {
            rack_id: float(
                required_flows[index]
            )
            for index, rack_id
            in enumerate(rack_ids)
        }

        # -------------------------------------
        # OEM curve range validation
        # -------------------------------------
        if rack_dp_curve is not None:
            invalid_required = [
                rack_id
                for rack_id in rack_ids
                if (
                    required_flow_map[rack_id]
                    < rack_dp_curve.q_min_lpm
                    or required_flow_map[rack_id]
                    > rack_dp_curve.q_max_lpm
                )
            ]

            if invalid_required:
                raise ValueError(
                    "Thermal required flow for one or more racks "
                    "is outside the verified OEM rack Q–ΔP range. "
                    "Detailed flow-distribution solving is blocked "
                    "to avoid extrapolation. "
                    f"Example rack: {invalid_required[0]}"
                )

        # =====================================
        # Rack internal ΔP model
        # =====================================
        def rack_internal_dp_kpa(
            rack_id: str,
            flow_lpm: float,
        ) -> float:
            flow = float(flow_lpm)

            if flow <= 0:
                return 0.0

            if rack_dp_curve is not None:
                return float(
                    rack_dp_curve.a
                    * flow**2
                    + rack_dp_curve.b
                    * flow
                )

            if water_ref_flow_lpm > 0:
                return float(
                    geom.rack_dp_reference_kpa
                    * (
                        coolant.rho_kg_m3
                        / water_reference.rho_kg_m3
                    )
                    * (
                        flow
                        / water_ref_flow_lpm
                    ) ** 2
                )

            return float(
                geom.rack_dp_reference_kpa
            )

        # =====================================
        # Evaluate complete hydraulic state
        # for a trial rack-flow vector
        # =====================================
        def evaluate_state(
            flow_vector,
        ):
            flow_map = {
                rack_id: float(
                    flow_vector[
                        rack_index[
                            rack_id
                        ]
                    ]
                )
                for rack_id in rack_ids
            }

            segment_state = {}

            for segment_number, segment in (
                pod_segments.iterrows()
            ):
                downstream_ids = segment[
                    "downstream_rack_ids"
                ]

                if isinstance(
                    downstream_ids,
                    str,
                ):
                    downstream_ids = (
                        downstream_ids,
                    )

                segment_flow = float(
                    sum(
                        flow_map.get(
                            str(rack_id),
                            0.0,
                        )
                        for rack_id
                        in downstream_ids
                    )
                )

                diameter_m = float(
                    segment[
                        "diameter_m"
                    ]
                )

                length_m = float(
                    segment[
                        "length_m"
                    ]
                )

                minor_k = float(
                    segment[
                        "minor_k"
                    ]
                )

                if segment_flow > 0:
                    pipe_dp = _pipe_dp_kpa(
                        segment_flow,
                        coolant.rho_kg_m3,
                        coolant.mu_pa_s,
                        length_m,
                        diameter_m,
                        geom.roughness_m,
                    )

                    minor_dp = _minor_dp_kpa(
                        segment_flow,
                        coolant.rho_kg_m3,
                        diameter_m,
                        minor_k,
                    )

                    velocity = _velocity(
                        segment_flow,
                        diameter_m,
                    )

                else:
                    pipe_dp = 0.0
                    minor_dp = 0.0
                    velocity = 0.0

                segment_state[
                    segment_number
                ] = {
                    "flow_lpm": segment_flow,
                    "velocity_m_s": velocity,
                    "pipe_dp_kpa": pipe_dp,
                    "minor_dp_kpa": minor_dp,
                    "total_dp_kpa": (
                        pipe_dp
                        + minor_dp
                    ),
                }

            total_pod_flow = float(
                sum(
                    flow_map.values()
                )
            )

        if topology_mode == "in_row":
            shared_common_minor_dp = 0.0

        else:
            shared_common_minor_dp = (
                _minor_dp_kpa(
                    total_pod_flow,
                    coolant.rho_kg_m3,
                    geom.common_diameter_m,
                    geom.common_minor_k,
                )
                if total_pod_flow > 0
                else 0.0
            )

            rack_path_state = {}

            for rack_id in rack_ids:
                common_header_dp = 0.0
                row_header_dp = 0.0
                branch_dp = 0.0

                for segment_number, segment in (
                    pod_segments.iterrows()
                ):
                    downstream_ids = segment[
                        "downstream_rack_ids"
                    ]

                    if isinstance(
                        downstream_ids,
                        str,
                    ):
                        downstream_ids = (
                            downstream_ids,
                        )

                    if rack_id not in [
                        str(item)
                        for item in downstream_ids
                    ]:
                        continue

                    segment_dp = float(
                        segment_state[
                            segment_number
                        ][
                            "total_dp_kpa"
                        ]
                    )

                    segment_type = str(
                        segment[
                            "segment_type"
                        ]
                    )

                    if (
                        segment_type
                        == "common_header"
                    ):
                        common_header_dp += (
                            segment_dp
                        )

                    elif (
                        segment_type
                        == "row_header"
                    ):
                        row_header_dp += (
                            segment_dp
                        )

                    elif (
                        segment_type
                        == "rack_branch_equivalent"
                    ):
                        branch_dp += (
                            segment_dp
                        )

                rack_dp = (
                    rack_internal_dp_kpa(
                        rack_id,
                        flow_map[
                            rack_id
                        ],
                    )
                )

                total_path_dp = (
                    common_header_dp
                    + row_header_dp
                    + branch_dp
                    + shared_common_minor_dp
                    + rack_dp
                )

                rack_path_state[
                    rack_id
                ] = {
                    "common_header_dp_kpa":
                        common_header_dp,
                    "row_header_dp_kpa":
                        row_header_dp,
                    "branch_dp_kpa":
                        branch_dp,
                    "shared_common_minor_dp_kpa":
                        shared_common_minor_dp,
                    "rack_dp_kpa":
                        rack_dp,
                    "total_path_dp_kpa":
                        total_path_dp,
                }

            return {
                "flow_map": flow_map,
                "segment_state": segment_state,
                "rack_path_state": rack_path_state,
                "total_flow_lpm": total_pod_flow,
            }

        # =====================================
        # Initial condition
        # =====================================
        initial_flows = (
            required_flows.copy()
        )

        initial_state = evaluate_state(
            initial_flows
        )

        initial_path_dps = [
            initial_state[
                "rack_path_state"
            ][rack_id][
                "total_path_dp_kpa"
            ]
            for rack_id in rack_ids
        ]

        initial_head = max(
            float(
                np.mean(
                    initial_path_dps
                )
            ),
            1.0,
        )

        # Unknown vector:
        # [Q1, Q2, ... Qn, common_head]
        initial_vector = np.concatenate(
            [
                initial_flows,
                np.array(
                    [
                        initial_head
                    ],
                    dtype=float,
                ),
            ]
        )

        pressure_scale = max(
            initial_head,
            1.0,
        )

        flow_scale = max(
            target_total_flow,
            1.0,
        )

        # =====================================
        # Bounds
        # =====================================
        if rack_dp_curve is not None:
            flow_lower_bounds = np.full(
                len(rack_ids),
                rack_dp_curve.q_min_lpm,
                dtype=float,
            )

            flow_upper_bounds = np.full(
                len(rack_ids),
                rack_dp_curve.q_max_lpm,
                dtype=float,
            )

            if (
                target_total_flow
                < float(
                    flow_lower_bounds.sum()
                )
                or target_total_flow
                > float(
                    flow_upper_bounds.sum()
                )
            ):
                raise ValueError(
                    "Total thermal required flow cannot be "
                    "distributed within the verified OEM "
                    "rack Q–ΔP flow range."
                )

        else:
            flow_lower_bounds = np.full(
                len(rack_ids),
                1e-6,
                dtype=float,
            )

            flow_upper_bounds = np.full(
                len(rack_ids),
                max(
                    target_total_flow,
                    float(
                        required_flows.max()
                        * 5.0
                    ),
                    1000.0,
                ),
                dtype=float,
            )

        lower_bounds = np.concatenate(
            [
                flow_lower_bounds,
                np.array(
                    [
                        1e-6
                    ],
                    dtype=float,
                ),
            ]
        )

        upper_bounds = np.concatenate(
            [
                flow_upper_bounds,
                np.array(
                    [
                        np.inf
                    ],
                    dtype=float,
                ),
            ]
        )

        # =====================================
        # Coupled nonlinear residuals
        # =====================================
        def residual_function(
            unknown_vector,
        ):
            rack_flows = (
                unknown_vector[
                    :-1
                ]
            )

            common_head = float(
                unknown_vector[
                    -1
                ]
            )

            state = evaluate_state(
                rack_flows
            )

            residuals = []

            # Every parallel rack path must see
            # the same hydraulic head.
            for rack_id in rack_ids:
                rack_path_dp = float(
                    state[
                        "rack_path_state"
                    ][rack_id][
                        "total_path_dp_kpa"
                    ]
                )

                residuals.append(
                    (
                        rack_path_dp
                        - common_head
                    )
                    / pressure_scale
                )

            # Preserve total thermal design flow
            residuals.append(
                (
                    float(
                        rack_flows.sum()
                    )
                    - target_total_flow
                )
                / flow_scale
            )

            return np.array(
                residuals,
                dtype=float,
            )

        solution = least_squares(
            residual_function,
            initial_vector,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            max_nfev=2000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        if not solution.success:
            raise ValueError(
                "Detailed hydraulic flow-distribution "
                "solver did not converge. "
                f"Solver message: {solution.message}"
            )

        solved_flows = (
            solution.x[
                :-1
            ]
        )

        solved_head = float(
            solution.x[
                -1
            ]
        )

        solved_state = evaluate_state(
            solved_flows
        )

        # Pump sizing basis includes reserve margin,
        # but the margin is not treated as a valve
        # resistance affecting distribution.
        pump_head_basis_kpa = (
            solved_head
            + max(
                float(
                    geom.balancing_margin_kpa
                ),
                0.0,
            )
        )

        eta = max(
            geom.pump_efficiency
            * geom.motor_efficiency,
            1e-6,
        )

        pump_power_kw = (
            pump_head_basis_kpa
            * 1000.0
            * (
                solved_state[
                    "total_flow_lpm"
                ]
                / 60000.0
            )
            / 1000.0
            / eta
        )

        # =====================================
        # Rack-level outputs
        # =====================================
        for _, rack_row in (
            pod_racks.iterrows()
        ):
            rack_id = str(
                rack_row[
                    "rack_id"
                ]
            )

            actual_flow = float(
                solved_state[
                    "flow_map"
                ][rack_id]
            )

            required_flow = float(
                rack_row[
                    "required_flow_lpm"
                ]
            )

            flow_margin_lpm = (
                actual_flow
                - required_flow
            )

            flow_margin_pct = (
                flow_margin_lpm
                / required_flow
                * 100.0
                if required_flow > 0
                else 0.0
            )

            path_state = solved_state[
                "rack_path_state"
            ][rack_id]

            rack_output_records.append(
                {
                    "coolant":
                        coolant.name,

                    "topology_mode":
                        topology_mode,

                    "hydraulic_group":
                        str(
                            hydraulic_group
                        ),

                    "pod":
                        str(
                            rack_row[
                                "pod"
                            ]
                        ),

                    "rack_id":
                        rack_id,
                    "row":
                        rack_row["row"],
                    "col":
                        rack_row["col"],
                    "required_flow_lpm":
                        required_flow,
                    "actual_flow_lpm":
                        actual_flow,
                    "flow_margin_lpm":
                        flow_margin_lpm,
                    "flow_margin_pct":
                        flow_margin_pct,
                    "underfed":
                        actual_flow
                        < required_flow,
                    "common_header_dp_kpa":
                        path_state[
                            "common_header_dp_kpa"
                        ],
                    "row_header_dp_kpa":
                        path_state[
                            "row_header_dp_kpa"
                        ],
                    "branch_dp_kpa":
                        path_state[
                            "branch_dp_kpa"
                        ],
                    "rack_dp_kpa":
                        path_state[
                            "rack_dp_kpa"
                        ],
                    "total_path_dp_kpa":
                        path_state[
                            "total_path_dp_kpa"
                        ],
                    "solved_pump_head_kpa":
                        solved_head,
                    "pump_head_basis_kpa":
                        pump_head_basis_kpa,
                }
            )

        # =====================================
        # Segment-level outputs
        # =====================================
        for segment_number, segment in (
            pod_segments.iterrows()
        ):
            segment_state = solved_state[
                "segment_state"
            ][segment_number]

            segment_output_records.append(
                {
                    "coolant":
                        coolant.name,

                    "topology_mode":
                        topology_mode,

                    "hydraulic_group":
                        str(
                            hydraulic_group
                        ),

                    "pod":
                        str(
                            segment[
                                "pod"
                            ]
                        ),

                    "segment_id":
                        ],
                    "segment_type":
                        segment[
                            "segment_type"
                        ],
                    "side":
                        segment[
                            "side"
                        ],
                    "row":
                        segment[
                            "row"
                        ],
                    "length_m":
                        segment[
                            "length_m"
                        ],
                    "diameter_m":
                        segment[
                            "diameter_m"
                        ],
                    "actual_flow_lpm":
                        segment_state[
                            "flow_lpm"
                        ],
                    "velocity_m_s":
                        segment_state[
                            "velocity_m_s"
                        ],
                    "pipe_dp_kpa":
                        segment_state[
                            "pipe_dp_kpa"
                        ],
                    "minor_dp_kpa":
                        segment_state[
                            "minor_dp_kpa"
                        ],
                    "total_dp_kpa":
                        segment_state[
                            "total_dp_kpa"
                        ],
                }
            )

        actual_flow_array = np.array(
            [
                solved_state[
                    "flow_map"
                ][rack_id]
                for rack_id in rack_ids
            ],
            dtype=float,
        )

        flow_margin_pct_array = (
            (
                actual_flow_array
                - required_flows
            )
            / required_flows
            * 100.0
        )

        pod_output_records.append(
            {
                "coolant":
                    coolant.name,

                "topology_mode":
                    topology_mode,

                "hydraulic_group":
                    str(
                        hydraulic_group
                    ),

                "pod":
                    summary_pod_label,

                "rack_count":
                    len(rack_ids),
                "required_total_flow_lpm":
                    target_total_flow,
                "actual_total_flow_lpm":
                    solved_state[
                        "total_flow_lpm"
                    ],
                "minimum_flow_margin_pct":
                    float(
                        flow_margin_pct_array.min()
                    ),
                "maximum_flow_margin_pct":
                    float(
                        flow_margin_pct_array.max()
                    ),
                "underfed_rack_count":
                    int(
                        (
                            actual_flow_array
                            < required_flows
                        ).sum()
                    ),
                "solved_pump_head_kpa":
                    solved_head,
                "balancing_margin_kpa":
                    max(
                        float(
                            geom.balancing_margin_kpa
                        ),
                        0.0,
                    ),
                "pump_head_basis_kpa":
                    pump_head_basis_kpa,
                "pump_power_kw":
                    pump_power_kw,
                "solver_converged":
                    bool(
                        solution.success
                    ),
            }
        )

    return {
        "rack_results": pd.DataFrame(
            rack_output_records
        ),
        "segment_results": pd.DataFrame(
            segment_output_records
        ),
        "pod_summary": pd.DataFrame(
            pod_output_records
        ),
    }

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
