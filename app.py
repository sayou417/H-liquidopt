from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from engine import (
    Coolant,
    HydraulicGeometry,
    heat_loads,
    pod_summary,
    evaluate_coolants,
    candidate_score_table,
    validate_racks,
    recommended_duty_cdus,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="H-LiquidOpt", page_icon="💧", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.25rem; max-width: 1500px;}
[data-testid="stMetric"] {border:1px solid #E3E7EC; border-radius:12px; padding:12px; background:#FFFFFF;}
.phase-card {border:1px solid #DFE4EA;border-radius:14px;padding:16px 18px;background:#FAFBFC;margin:8px 0 14px 0;}
.phase-ok {border-left:5px solid #2E7D32;}
.phase-warn {border-left:5px solid #B7791F;}
.small-note {font-size:0.88rem;color:#616B75;}
.section-label {font-weight:700;color:#243447;margin-bottom:0.25rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_racks() -> pd.DataFrame:
    return pd.read_csv(DATA / "sample_48_racks.csv")


@st.cache_data
def load_default_coolants() -> pd.DataFrame:
    return pd.read_csv(DATA / "coolants.csv")


def reset_downstream(from_phase: int = 1) -> None:
    for p in range(from_phase, 5):
        st.session_state.approved[p] = False
    if from_phase <= 2:
        st.session_state.pop("topology_choice", None)
    if from_phase <= 3:
        st.session_state.pop("coolant_names", None)
    if from_phase <= 4:
        st.session_state.pop("hydraulic_results", None)


def project_report_markdown(racks, pods, ranking, names, delta_t, cdu_capacity, redundancy) -> str:
    calc = heat_loads(racks)
    best = ranking.iloc[0] if len(ranking) else None
    lines = [
        "# H-LiquidOpt Prototype Design Review",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Project summary",
        f"- Rack count: {len(calc)}",
        f"- Pod count: {calc['pod'].nunique()}",
        f"- Total IT load: {calc['it_power_kw'].sum()/1000:.3f} MW",
        f"- Liquid design load: {calc['liquid_load_kw'].sum()/1000:.3f} MW",
        f"- Residual air load: {calc['residual_air_kw'].sum()/1000:.3f} MW",
        f"- Design delta-T: {delta_t:.1f} K",
        f"- CDU candidate capacity: {cdu_capacity:.2f} MW/unit",
        f"- Redundancy selection: {redundancy}",
        "",
        "## Engineer decisions",
        f"- Phase 1: {'Approved' if st.session_state.approved[1] else 'Pending'}",
        f"- Phase 2 topology: {st.session_state.get('topology_choice', 'Not selected')}",
        f"- Phase 3 coolant sensitivity candidates: {', '.join(names)}",
        f"- Phase 4: {'Reviewed' if st.session_state.approved[4] else 'Pending'}",
        "",
        "## Pod loads",
        pods.to_markdown(index=False),
        "",
        "## Candidate comparison",
        ranking[["rank","coolant","total_pump_kw","worst_dp_kpa","cdu_loading_pct","balanced_score"]].to_markdown(index=False) if len(ranking) else "No candidate result.",
        "",
    ]
    if best is not None:
        lines += [
            "## Current PoC sensitivity result",
            f"- Lowest balanced hydraulic sensitivity candidate: {best['coolant']}",
            f"- Total pump power: {best['total_pump_kw']:.2f} kW",
            f"- Worst calculated pressure drop: {best['worst_dp_kpa']:.1f} kPa",
            "",
        ]
    lines += [
        "## Limitations",
        "- This prototype is not for construction, procurement, safety certification, or final equipment/coolant selection.",
        "- Rack internal pressure drop uses a synthetic placeholder unless replaced with an OEM pressure-flow curve.",
        "- Coolant candidates require OEM/supplier approval, material compatibility, water chemistry and freeze-condition verification.",
        "- The heat-load map is a load-density visualization, not CFD temperature prediction.",
    ]
    return "\n".join(lines)


if "racks" not in st.session_state:
    st.session_state.racks = load_default_racks()
if "approved" not in st.session_state:
    st.session_state.approved = {1: False, 2: False, 3: False, 4: False}

# Persistent project inputs
defaults = {
    "supply_t": 35.0,
    "return_t": 45.0,
    "cdu_capacity": 2.0,
    "redundancy": "N+1 shared standby",
    "common_d": 0.2027,
    "row_d": 0.1541,
    "branch_d": 0.0525,
    "common_l": 20.0,
    "row_l": 12.0,
    "branch_l": 6.0,
    "rack_dp": 120.0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def render_tag(text, kind="info"):
    styles = {
        "verified": ("#EAF7EE", "#137333", "#B7DFC2"),
        "input": ("#EAF2FF", "#2457A6", "#BFD1F2"),
        "assumption": ("#FFF4E5", "#A85D00", "#F1D19A"),
        "review": ("#FDECEC", "#B3261E", "#F2B8B5"),
        "calculated": ("#F2ECFF", "#6542A6", "#D3C4F3"),
    }

    bg, fg, border = styles.get(kind, styles["input"])

    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:3px 9px;
            margin-right:5px;
            border-radius:12px;
            background:{bg};
            color:{fg};
            border:1px solid {border};
            font-size:11px;
            font-weight:600;
        ">
            {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

st.title("H-LiquidOpt")
st.caption("Human-in-the-Loop · D2C liquid-cooling preliminary design support prototype")
phase = st.radio(
    "Design phase",
    [1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "1 · Thermal",
        2: "2 · TCS / CDU",
        3: "3 · Coolant",
        4: "4 · Hydraulics",
        5: "5 · Final Review",
    }[x],
    horizontal=True,
    key="design_phase",
)

with st.sidebar:
    st.header("H-LiquidOpt")
    st.caption(f"Design workflow · Phase {phase} of 5")

    # -------------------------
    # PHASE 1
    # -------------------------
    if phase == 1:
        st.subheader("Rack / IT Inputs")

        st.caption(
            "Upload the project rack dataset or use the built-in demo case."
        )

        template = pd.DataFrame({
            "rack_id": ["R001", "R002"],
            "pod": ["A", "A"],
            "rack_type": ["Compute", "Support"],
            "it_power_kw": [120.0, 20.0],
            "hcr": [0.85, 0.0],
            "row": [1, 1],
            "col": [1, 2],
        })

        st.download_button(
            "Download Rack CSV template",
            template.to_csv(index=False).encode("utf-8-sig"),
            "hliquidopt_rack_template.csv",
            "text/csv",
            use_container_width=True,
        )

        uploaded = st.file_uploader(
            "Upload Rack CSV",
            type=["csv"],
            key="rack_csv_upload",
        )

        if uploaded is not None:
            new_racks = pd.read_csv(uploaded)

            if not new_racks.equals(st.session_state.racks):
                st.session_state.racks = new_racks
                reset_downstream(1)

        if st.button(
            "Restore 48-rack demo",
            use_container_width=True,
        ):
            st.session_state.racks = load_default_racks()
            reset_downstream(1)
            st.rerun()

        st.divider()

        st.caption(
            "Phase 1 only requires rack and spatial data. "
            "Hydraulic parameters are entered later."
        )

    # -------------------------
    # PHASE 2
    # -------------------------
    elif phase == 2:
        st.subheader("TCS / CDU Inputs")

        st.number_input(
            "CDU Candidate Capacity (MW/unit)",
            min_value=0.1,
            step=0.1,
            key="cdu_capacity",
        )

        st.selectbox(
            "Redundancy",
            [
                "N+1 shared standby",
                "N",
                "2N",
            ],
            key="redundancy",
        )

        st.divider()

        st.caption(
            "These values are used to screen TCS/CDU topology candidates."
        )

    # -------------------------
    # PHASE 3
    # -------------------------
    elif phase == 3:
        st.subheader("Coolant Conditions")

        st.number_input(
            "TCS Supply Temperature (°C)",
            step=1.0,
            key="supply_t",
        )

        st.number_input(
            "TCS Return Temperature (°C)",
            step=1.0,
            key="return_t",
        )

        current_dt = (
            st.session_state.return_t
            - st.session_state.supply_t
        )

        st.metric(
            "Design ΔT",
            f"{current_dt:.1f} K",
        )

        if current_dt <= 0:
            st.error(
                "Return temperature must be greater than supply temperature."
            )

        st.divider()

        st.caption(
            "Coolant selection remains subject to OEM and supplier validation."
        )

    # -------------------------
    # PHASE 4
    # -------------------------
    elif phase == 4:
        st.subheader("Hydraulic Inputs")

        st.markdown("**Pipe geometry**")

        st.number_input(
            "Common pipe ID (m)",
            min_value=0.001,
            format="%.4f",
            key="common_d",
        )

        st.number_input(
            "Row header ID (m)",
            min_value=0.001,
            format="%.4f",
            key="row_d",
        )

        st.number_input(
            "Rack branch ID (m)",
            min_value=0.001,
            format="%.4f",
            key="branch_d",
        )

        st.markdown("**Equivalent supply + return length**")

        st.number_input(
            "Common pipe length (m)",
            min_value=0.0,
            step=1.0,
            key="common_l",
        )

        st.number_input(
            "Row header length (m)",
            min_value=0.0,
            step=1.0,
            key="row_l",
        )

        st.number_input(
            "Rack branch length (m)",
            min_value=0.0,
            step=1.0,
            key="branch_l",
        )

        st.divider()

        st.number_input(
            "Synthetic rack ΔP reference (kPa)",
            min_value=0.0,
            step=5.0,
            key="rack_dp",
            help=(
                "PoC placeholder only. "
                "Final engineering use requires an OEM pressure-flow curve."
            ),
        )

        st.warning(
            "Rack ΔP is currently an ASSUMPTION used for prototype sensitivity."
        )

    # -------------------------
    # PHASE 5
    # -------------------------
    else:
        st.subheader("Final Review")

        st.caption(
            "No new engineering inputs are required in this phase."
        )

        st.metric(
            "Approved phases",
            f"{sum(bool(st.session_state.approved[p]) for p in [1,2,3,4])} / 4",
        )


# Values persist even when their input widgets are hidden
supply_t = float(st.session_state.supply_t)
return_t = float(st.session_state.return_t)
delta_t = return_t - supply_t

cdu_capacity = float(st.session_state.cdu_capacity)
redundancy = st.session_state.redundancy

common_d = float(st.session_state.common_d)
row_d = float(st.session_state.row_d)
branch_d = float(st.session_state.branch_d)

common_l = float(st.session_state.common_l)
row_l = float(st.session_state.row_l)
branch_l = float(st.session_state.branch_l)

rack_dp = float(st.session_state.rack_dp)


geom = HydraulicGeometry(
    common_length_m=common_l,
    common_diameter_m=common_d,
    row_length_m=row_l,
    row_diameter_m=row_d,
    branch_length_m=branch_l,
    branch_diameter_m=branch_d,
    rack_dp_reference_kpa=rack_dp,
)

errors = validate_racks(st.session_state.racks)
if errors:
    st.error("Rack input validation failed:\n- " + "\n- ".join(errors))
    st.stop()

calc_now = heat_loads(st.session_state.racks)
progress_count = sum(bool(st.session_state.approved[p]) for p in [1,2,3,4])
st.progress(progress_count / 4, text=f"Engineer review progress · {progress_count}/4 phases approved")

hero1, hero2, hero3, hero4 = st.columns(4)
hero1.metric("Rack count", f"{len(calc_now)}")
hero2.metric("Pods", f"{calc_now['pod'].nunique()}")
hero3.metric("IT load", f"{calc_now['it_power_kw'].sum()/1000:.2f} MW")
hero4.metric("Liquid load", f"{calc_now['liquid_load_kw'].sum()/1000:.2f} MW")

if phase == 1:
    st.header("Phase 1 · Rack Heat Load & Spatial Review")

    st.caption(
        "Rack 및 IT 부하 정보를 확인하고, "
        "Liquid / Residual Air Heat Load와 공간별 부하 분포를 검토합니다."
    )

    # -----------------------------------
    # 1A. INPUT
    # -----------------------------------
    st.markdown("### 1A · Input Data")
    input_mode = st.radio(
        "Rack input mode",
        [
            "Quick Rack Input",
            "Detailed Equipment Input",
        ],
        horizontal=True,
        key="rack_input_mode",
    )

    if input_mode == "Quick Rack Input":
        st.caption(
            "Rack별 IT Power가 이미 산정된 경우 사용합니다. "
            "Rack Power, HCR, Pod 및 위치정보를 직접 입력합니다."
        )
        
        active_racks = st.session_state.racks
    else:
        st.caption(
            "Rack 내부 장비 구성과 수량을 입력하면 "
            "장비별 설계전력을 합산하여 Rack Power를 자동 산정합니다."
        )
        # -----------------------------------
        # Detailed Equipment Input
        # -----------------------------------
        if "equipment_input" not in st.session_state:
            st.session_state.equipment_input = pd.DataFrame(
                [
                    {
                        "rack_id": "R01",
                        "equipment": "GPU Server",
                        "quantity": 8,
                        "rated_power_kw": 10.2,
                        "load_factor": 1.00,
                        "hcr": 0.85,
                        "pod": "A",
                        "row": 1,
                        "col": 1,
                    },
                    {
                        "rack_id": "R01",
                        "equipment": "Network Switch",
                        "quantity": 2,
                        "rated_power_kw": 1.0,
                        "load_factor": 1.00,
                        "hcr": 0.85,
                        "pod": "A",
                        "row": 1,
                        "col": 1,
                    },
                ]
            )

        # -----------------------------------
        # Equipment Configuration
        # -----------------------------------
        st.markdown("#### Equipment Configuration")

        equipment_columns = [
            "rack_id",
            "equipment",
            "quantity",
            "rated_power_kw",
            "load_factor",
            "hcr",
        ]

        equipment_for_editor = st.session_state.equipment_input.copy()

        # 기존 session에 pod / row / col 등이 남아 있어도
        # Equipment 표에서는 장비 관련 열만 사용
        equipment_for_editor = equipment_for_editor[
            equipment_columns
        ]

        edited_equipment = st.data_editor(
            equipment_for_editor,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "quantity": st.column_config.NumberColumn(
                    "Qty",
                    min_value=0,
                    step=1,
                ),
                "rated_power_kw": st.column_config.NumberColumn(
                    "Rated Power (kW/unit)",
                    min_value=0.0,
                    step=0.1,
                    format="%.2f",
                ),
                "load_factor": st.column_config.NumberColumn(
                    "Load Factor",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format="%.2f",
                ),
                "hcr": st.column_config.NumberColumn(
                    "HCR",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                ),
            },
            key="equipment_editor",
        )

        equipment_calc = edited_equipment.copy()

        # -----------------------------------
        # Rack Metadata
        # -----------------------------------
        st.markdown("#### Rack Metadata")

        st.caption(
            "Rack Type, Pod 및 배치정보는 장비별이 아니라 Rack별로 한 번만 지정합니다."
        )

        current_rack_ids = (
            edited_equipment["rack_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        current_rack_ids = [
            rack_id
            for rack_id in current_rack_ids.unique().tolist()
            if rack_id
        ]

        if "rack_metadata" not in st.session_state:
            st.session_state.rack_metadata = pd.DataFrame(
                columns=[
                    "rack_id",
                    "rack_type",
                    "pod",
                    "row",
                    "col",
                ]
            )

        metadata = st.session_state.rack_metadata.copy()

        # Equipment 표에 새 Rack ID가 생기면 Metadata에도 자동 추가
        existing_ids = (
            metadata["rack_id"].astype(str).tolist()
            if not metadata.empty
            else []
        )

        new_rows = []

        for rack_id in current_rack_ids:
            if rack_id not in existing_ids:
                new_rows.append(
                    {
                        "rack_id": rack_id,
                        "rack_type": "compute",
                        "pod": "A",
                        "row": 1,
                        "col": 1,
                    }
                )

        if new_rows:
            metadata = pd.concat(
                [
                    metadata,
                    pd.DataFrame(new_rows),
                ],
                ignore_index=True,
            )

        # 현재 Equipment에 존재하는 Rack만 표시
        metadata = metadata[
            metadata["rack_id"].astype(str).isin(
                current_rack_ids
            )
        ].copy()

        edited_metadata = st.data_editor(
            metadata,
            use_container_width=True,
            hide_index=True,
            column_config={
                "rack_id": st.column_config.TextColumn(
                    "Rack ID",
                    disabled=True,
                ),
                "rack_type": st.column_config.SelectboxColumn(
                    "Rack Type",
                    options=[
                        "compute",
                        "support",
                        "network",
                        "storage",
                        "other",
                    ],
                ),
                "pod": st.column_config.TextColumn(
                    "Pod",
                ),
                "row": st.column_config.NumberColumn(
                    "Row",
                    min_value=1,
                    step=1,
                ),
                "col": st.column_config.NumberColumn(
                    "Column",
                    min_value=1,
                    step=1,
                ),
            },
            key="rack_metadata_editor",
        )

        equipment_calc = edited_equipment.copy()

        equipment_calc["design_power_kw"] = (
            pd.to_numeric(
                equipment_calc["quantity"],
                errors="coerce",
            ).fillna(0)
            * pd.to_numeric(
                equipment_calc["rated_power_kw"],
                errors="coerce",
            ).fillna(0)
            * pd.to_numeric(
                equipment_calc["load_factor"],
                errors="coerce",
            ).fillna(0)
        )

        rack_power_summary = (
            equipment_calc
            .groupby("rack_id", as_index=False)["design_power_kw"]
            .sum()
            .rename(
                columns={
                    "design_power_kw": "rack_power_kw"
                }
            )
        )

        st.markdown("#### Calculated Rack Power")

        st.dataframe(
            rack_power_summary,
            use_container_width=True,
            hide_index=True,
        )
        # -----------------------------------
        # Convert equipment data to rack-level input
        # -----------------------------------

        equipment_calc["hcr_numeric"] = pd.to_numeric(
            equipment_calc["hcr"],
            errors="coerce",
        ).fillna(0)

        equipment_calc["liquid_design_kw"] = (
            equipment_calc["design_power_kw"]
            * equipment_calc["hcr_numeric"]
        )

        # -----------------------------------
        # Aggregate equipment → Rack power
        # -----------------------------------
        rack_level = (
            equipment_calc
            .groupby("rack_id", as_index=False)
            .agg(
                it_power_kw=("design_power_kw", "sum"),
                liquid_design_kw=("liquid_design_kw", "sum"),
            )
        )
        # Effective Rack HCR
        rack_level["hcr"] = rack_level.apply(
            lambda x: (
                x["liquid_design_kw"] / x["it_power_kw"]
                if x["it_power_kw"] > 0
                else 0.0
            ),
            axis=1,
        )

        # -----------------------------------
        # Merge Rack Metadata
        # -----------------------------------
        rack_level = rack_level.merge(
            edited_metadata[
                [
                    "rack_id",
                    "rack_type",
                    "pod",
                    "row",
                    "col",
                ]
            ],
            on="rack_id",
            how="left",
        )

        rack_level["hcr"] = rack_level.apply(
            lambda x: (
                x["liquid_design_kw"] / x["it_power_kw"]
                if x["it_power_kw"] > 0
                else 0.0
            ),
            axis=1,
        )

        detailed_racks = rack_level[
            [
                "rack_id",
                "rack_type",
                "pod",
                "it_power_kw",
                "hcr",
                "row",
                "col",
            ]
        ].copy()

        active_racks = detailed_racks
       
    c1, c2 = st.columns([1, 4])

    with c1:
        render_tag("USER INPUT", "input")

    with c2:
        st.write(
            "Rack CSV 또는 데모 데이터를 기반으로 분석합니다. "
            "Rack 수, IT Power, HCR, Pod 및 위치정보를 엔지니어가 검토할 수 있습니다."
        )

    racks = active_racks

    rack_count = len(racks)
    pod_count = racks["pod"].nunique()

    total_it_mw = racks["it_power_kw"].sum() / 1000.0
    liquid_load_mw = (
        racks["it_power_kw"] * racks["hcr"]
    ).sum() / 1000.0

    residual_air_mw = total_it_mw - liquid_load_mw

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Rack Count", f"{rack_count}")

    with m2:
        st.metric("Cooling Pods", f"{pod_count}")

    with m3:
        st.metric("Total IT Load", f"{total_it_mw:.2f} MW")

    with m4:
        st.metric("Liquid Load", f"{liquid_load_mw:.2f} MW")

    st.markdown("**Input provenance**")

    t1, t2, t3 = st.columns(3)

    with t1:
        render_tag("USER INPUT", "input")
        st.caption("Rack 수 / 배치 / IT Power")

    with t2:
        render_tag("USER INPUT", "input")
        st.caption("Heat Capture Ratio")

    with t3:
        render_tag("CALCULATED", "calculated")
        st.caption("Liquid / Residual Air Load")

    st.markdown("#### Rack Configuration")

    if input_mode == "Quick Rack Input":

        st.info(
            "Quick Input에서는 Rack-level 값을 직접 수정할 수 있습니다. "
            "표의 값을 수정하면 Phase 1 계산 결과가 즉시 갱신됩니다."
        )

        edited_racks = st.data_editor(
            st.session_state.racks,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "hcr": st.column_config.NumberColumn(
                    "HCR",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                ),
                "it_power_kw": st.column_config.NumberColumn(
                    "IT Power (kW)",
                    min_value=0.0,
                    step=5.0,
                ),
            },
            key="rack_editor_quick",
        )

        if not edited_racks.equals(st.session_state.racks):
            st.session_state.racks = edited_racks
            reset_downstream(1)

    else:

        st.info(
            "Detailed Equipment Input의 장비 구성으로부터 자동 생성된 "
            "Rack-level 설계 데이터입니다. Rack Power는 직접 입력하지 않습니다."
        )

        edited_racks = active_racks.copy()

        st.dataframe(
            edited_racks,
            use_container_width=True,
            hide_index=True,
        )

    errors = validate_racks(edited_racks)

    if errors:
        st.error(
            "Fix rack data before approval:\n- "
            + "\n- ".join(errors)
        )
        st.stop()

    calc = heat_loads(edited_racks)
    pods = pod_summary(edited_racks)

    # -----------------------------------
    # 1B. ANALYSIS
    # -----------------------------------
    st.divider()

    st.markdown("### 1B · Analysis")

    a1, a2 = st.columns([1, 4])

    with a1:
        render_tag("CALCULATED", "calculated")

    with a2:
        st.write(
            "Rack별 IT Power와 HCR을 이용해 Liquid Heat Load와 "
            "Residual Air Heat Load를 계산하고 공간별 부하 분포를 분석합니다."
        )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Rack positions", len(calc))
    m2.metric(
        "Total IT Load",
        f"{calc['it_power_kw'].sum()/1000:.2f} MW",
    )
    m3.metric(
        "Liquid Load",
        f"{calc['liquid_load_kw'].sum()/1000:.2f} MW",
    )
    m4.metric(
        "Residual Air",
        f"{calc['residual_air_kw'].sum()/1000:.2f} MW",
    )

    left, right = st.columns([1.35, 1])

    with left:
        if {"row", "col"}.issubset(calc.columns):
            pivot = calc.pivot(
                index="row",
                columns="col",
                values="liquid_load_kw",
            )

            fig = px.imshow(
                pivot,
                text_auto=".0f",
                aspect="auto",
                labels={"color": "Liquid kW"},
                title="Rack Liquid Heat-Load Density Map",
                color_continuous_scale="Blues",
            )

            fig.update_xaxes(
                side="top",
                title="Column",
            )

            fig.update_yaxes(
                title="Row",
                autorange="reversed",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "※ 본 Heat-Load Density Map은 CFD 기반 실제 온도 Hot Spot이 아니라, "
                "Rack별 liquid-side 설계 열부하의 공간적 분포를 나타냅니다."
            )

        else:
            st.info(
                "No row/col columns: thermal totals still work, "
                "but spatial heat map is skipped."
            )

    with right:
        st.markdown("#### Pod summary")

        st.dataframe(
            pods,
            use_container_width=True,
            hide_index=True,
        )

        hottest = pods.loc[
            pods["liquid_load_kw"].idxmax()
        ]

        st.markdown(
            f"<div class='phase-card'>"
            f"Highest liquid-load pod: <b>{hottest['pod']}</b><br>"
            f"{hottest['liquid_load_kw']/1000:.3f} MW liquid load · "
            f"{int(hottest['liquid_racks'])} liquid-cooled racks"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Analysis Note")

    st.info(
        f"현재 입력조건에서는 **Pod {hottest['pod']}**의 Liquid Load가 "
        f"가장 높으며, 약 **{hottest['liquid_load_kw']/1000:.3f} MW**입니다. "
        "Phase 2에서는 해당 Pod의 CDU loading과 Zone별 설비 구성을 "
        "우선 검토할 수 있습니다."
    )

    # -----------------------------------
    # 1C. ENGINEER REVIEW
    # -----------------------------------
    st.divider()

    st.markdown("### 1C · Engineer Review")

    r1, r2 = st.columns([1, 4])

    with r1:
        render_tag(
            "REVIEW REQUIRED",
            "review",
        )

    with r2:
        st.write(
            "다음 단계로 진행하기 전에 Rack 입력조건과 "
            "열부하 분석결과를 엔지니어가 직접 확인합니다."
        )

    check_rack = st.checkbox(
        "Rack 구성 및 IT Power를 확인했습니다.",
        key="phase1_check_rack",
    )

    check_hcr = st.checkbox(
        "Heat Capture Ratio(HCR) 값 또는 적용 가정을 확인했습니다.",
        key="phase1_check_hcr",
    )

    check_pod = st.checkbox(
        "Pod 구성 및 Heat-Load Density Map을 확인했습니다.",
        key="phase1_check_pod",
    )

    phase1_note = st.text_area(
        "Engineer note",
        key="phase1_note",
        placeholder=(
            "예: Pod A의 Liquid Load가 가장 높음. "
            "Phase 2에서 Pod A의 CDU loading을 우선 검토."
        ),
    )

    ready_phase1 = (
        check_rack
        and check_hcr
        and check_pod
    )

    if st.button(
        "✓ Approve Phase 1",
        type="primary",
        disabled=not ready_phase1,
        use_container_width=True,
    ):
        # 승인된 Phase 1 설계 데이터를 Phase 2로 전달
        st.session_state.phase1_racks = edited_racks.copy()
        st.session_state.phase1_input_mode = input_mode

        st.session_state.approved[1] = True

        st.success(
            "Phase 1 approved. "
            "승인된 Rack Load Model이 Phase 2로 전달되었습니다."
        )

    if st.session_state.approved[1]:
        st.success(
            "✓ Phase 1 Engineer Review Approved"
        )

elif phase == 2:
    st.header("Phase 2 · TCS / CDU Candidate Review")

    st.caption(
        "Phase 1에서 승인된 Liquid Heat Load를 기반으로 "
        "TCS 구성과 CDU 배치 후보를 비교합니다."
    )

    if not st.session_state.approved[1]:
        st.warning(
            "Phase 1 Engineer Review가 아직 완료되지 않았습니다. "
            "Phase 1을 승인한 후 TCS / CDU 후보를 검토할 수 있습니다."
        )
        st.stop()

    if "phase1_racks" not in st.session_state:
        st.error(
            "승인된 Phase 1 Rack 데이터가 없습니다. "
            "Phase 1에서 Engineer Review를 다시 승인해주세요."
        )
        st.stop()

    phase1_racks = st.session_state.phase1_racks.copy()

    # -----------------------------------
    # 2A. DESIGN BASIS
    # -----------------------------------
    st.markdown("### 2A · Design Basis")

    b1, b2 = st.columns([1, 4])

    with b1:
        render_tag("PHASE 1 APPROVED", "verified")

    with b2:
        st.write(
            "Phase 1에서 엔지니어가 승인한 Rack Load Model을 기반으로 "
            "Pod별 Liquid Load와 CDU 용량 요구조건을 계산합니다."
        )

    pods = pod_summary(phase1_racks)

    pods["CDU loading %"] = (
        pods["liquid_load_kw"]
        / (cdu_capacity * 1000)
        * 100
    )

    total_liquid = pods["liquid_load_kw"].sum()

    min_duty = recommended_duty_cdus(
        total_liquid,
        cdu_capacity,
    )

    max_pod_loading = pods["CDU loading %"].max()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Total Liquid Load",
        f"{total_liquid / 1000:.2f} MW",
    )

    m2.metric(
        "Candidate CDU Capacity",
        f"{cdu_capacity:.1f} MW",
    )

    m3.metric(
        "Aggregate Duty Units",
        f"{min_duty}",
    )

    m4.metric(
        "Max Pod Loading",
        f"{max_pod_loading:.1f}%",
    )

    st.markdown("#### Pod Load Basis")

    st.dataframe(
        pods,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "※ Aggregate Duty Units는 총 Liquid Load를 CDU 정격용량으로 나눈 "
        "기초 용량 산정값입니다. 실제 CDU 수량은 Pod 분리, redundancy, "
        "유지보수 조건 및 배관 topology 검토에 따라 증가할 수 있습니다."
    )

    # -----------------------------------
    # 2B. CANDIDATE COMPARISON
    # -----------------------------------
    st.divider()

    st.markdown("### 2B · Candidate Comparison")

    c1, c2 = st.columns([1, 4])

    with c1:
        render_tag("CALCULATED", "calculated")

    with c2:
        st.write(
            "동일한 Liquid Load에 대해 Pod-dedicated, Central, "
            "In-row 구성의 기초 용량 적합성과 설비 구성을 비교합니다."
        )

    # Phase 1 rack data → calculated liquid load
    phase2_calc = heat_loads(phase1_racks)

    # -----------------------------------
    # Candidate A · Pod-dedicated
    # -----------------------------------
    import math

    cdu_capacity_kw = cdu_capacity * 1000

    pod_sizing = pods.copy()

    # 각 Pod 부하를 처리하기 위해 필요한 Duty CDU 수
    pod_sizing["Required Duty CDU"] = pod_sizing[
        "liquid_load_kw"
    ].apply(
        lambda load: (
            math.ceil(load / cdu_capacity_kw)
            if load > 0
            else 0
        )
    )

    # 동일 Pod 내 Duty CDU들이 부하를 균등 분담한다고 가정한 Loading
    pod_sizing["Allocated CDU Loading %"] = pod_sizing.apply(
        lambda row: (
            row["liquid_load_kw"]
            / (
                row["Required Duty CDU"]
                * cdu_capacity_kw
            )
            * 100
            if row["Required Duty CDU"] > 0
            else 0.0
        ),
        axis=1,
    )

    pod_duty_units = int(
        pod_sizing["Required Duty CDU"].sum()
    )

    pod_max_loading = (
        pod_sizing["Allocated CDU Loading %"].max()
    )

    pod_multi_cdu_required = (
        pod_sizing["Required Duty CDU"] > 1
    ).any()

    pod_capacity_ok = True

    # -----------------------------------
    # Candidate B · Central CDU plant
    # -----------------------------------
    central_duty_units = recommended_duty_cdus(
        total_liquid,
        cdu_capacity,
    )

    central_loading = (
        total_liquid
        / (central_duty_units * cdu_capacity * 1000)
        * 100
    )

    # -----------------------------------
    # Candidate C · In-row grouping
    # -----------------------------------
    if "row" in phase2_calc.columns:
        row_summary = (
            phase2_calc
            .groupby("row", as_index=False)["liquid_load_kw"]
            .sum()
        )

        row_count = len(row_summary)

        row_summary["Required Duty CDU"] = row_summary[
            "liquid_load_kw"
        ].apply(
            lambda load: (
                math.ceil(load / cdu_capacity_kw)
                if load > 0
                else 0
            )
        )

        row_summary["Allocated CDU Loading %"] = (
            row_summary.apply(
                lambda row: (
                    row["liquid_load_kw"]
                    / (
                        row["Required Duty CDU"]
                        * cdu_capacity_kw
                    )
                    * 100
                    if row["Required Duty CDU"] > 0
                    else 0.0
                ),
                axis=1,
            )
        )

        row_duty_units = int(
            row_summary["Required Duty CDU"].sum()
        )

        row_max_loading = (
            row_summary["Allocated CDU Loading %"].max()
        )

        row_multi_cdu_required = (
            row_summary["Required Duty CDU"] > 1
        ).any()

        row_capacity_ok = True

    else:
        row_summary = pd.DataFrame()
        row_count = 0
        row_duty_units = 0
        row_capacity_ok = False
        row_max_loading = 0.0
        row_multi_cdu_required = False

    # -----------------------------------
    # Candidate cards
    # -----------------------------------
    card_a, card_b, card_c = st.columns(3)

    with card_a:
        st.markdown("#### A · Pod-dedicated")

        st.success("Capacity sizing · FEASIBLE")

        st.metric(
            "Total Duty CDU Units",
            f"{pod_duty_units}",
        )

        st.metric(
            "Max Allocated Loading",
            f"{pod_max_loading:.1f}%",
        )

        if pod_multi_cdu_required:
            st.warning(
                "하나 이상의 Pod에서 2대 이상의 Duty CDU가 필요합니다."
            )
        with st.expander("Pod-level CDU sizing"):
            st.dataframe(
                pod_sizing[
                    [
                        "pod",
                        "liquid_load_kw",
                        "Required Duty CDU",
                        "Allocated CDU Loading %",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )    

        st.markdown(
            """
            **Concept**

            Pod별 독립 CDU를 배치하여
            Cooling Zone 경계를 명확하게 구성합니다.

            **Strength**
            - Pod 단위 격리 용이
            - 장애 영향 범위 제한
            - 단계별 증설에 유리

            **Review**
            - 각 Pod가 단일 CDU 정격 내에 들어오는지 확인
            - CDU 수량 증가 가능
            """
        )

    with card_b:
        st.markdown("#### B · Central CDU Plant")

        st.success("Aggregate capacity · PASS")

        st.metric(
            "Duty CDU Units",
            f"{central_duty_units}",
        )

        st.metric(
            "Average Duty Loading",
            f"{central_loading:.1f}%",
        )

        st.markdown(
            """
            **Concept**

            중앙 CDU Plant에서 여러 Pod의
            Liquid Load를 통합 처리합니다.

            **Strength**
            - 용량 Pooling 가능
            - 중앙 집중 유지보수
            - CDU 활용률 조정 용이

            **Review**
            - 배관 길이 증가 가능
            - Hydraulic balancing 검토 필요
            - 장애 영향 범위가 커질 수 있음
            """
        )

    with card_c:
        st.markdown("#### C · In-row Grouping")

        if row_capacity_ok:
            st.success("Capacity sizing · FEASIBLE")
        else:
            st.warning("Row data · REVIEW REQUIRED")

        st.metric(
            "Total Duty CDU Units",
            f"{row_duty_units}",
        )

        st.metric(
            "Max Allocated Loading",
            f"{row_max_loading:.1f}%",
        )

        if row_multi_cdu_required:
            st.warning(
                "하나 이상의 Row Group에서 2대 이상의 Duty CDU가 필요합니다."
            )

        if not row_summary.empty:
            with st.expander("Row-level CDU sizing"):
                st.dataframe(
                    row_summary,
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown(
            """
            **Concept**

            Row 또는 근접 Rack Group 단위로
            CDU를 부하 가까이에 배치합니다.

            **Strength**
            - Secondary loop 단축 가능
            - 부하 가까이에서 제어 가능
            - 구역별 확장에 유리

            **Review**
            - White-space 점유
            - 유지보수 동선
            - 실제 Row grouping 기준 검토 필요
            """
        )

    st.caption(
        "※ 본 비교는 기본설계 단계의 후보 스크리닝입니다. "
        "배관 Routing, 실제 CDU 성능곡선, 제어방식, 유지보수 공간 및 "
        "Redundancy 조건을 반영한 최종 설계 결과가 아닙니다."
    )

    topology_options = [
        "A · Pod-dedicated CDU",
        "B · Central CDU Plant",
        "C · In-row CDU Grouping",
    ]

    choice = st.radio(
        "Engineer-selected candidate",
        topology_options,
        horizontal=True,
        key="phase2_topology_choice",
    )
        # -----------------------------------
    # 2C. ENGINEER REVIEW
    # -----------------------------------
    st.divider()

    st.markdown("### 2C · Engineer Review")

    r1, r2 = st.columns([1, 4])

    with r1:
        render_tag("REVIEW REQUIRED", "review")

    with r2:
        st.write(
            "CDU 용량 적합성, redundancy 구성, 유지보수성과 "
            "배관 topology 검토사항을 확인한 후 후보안을 승인합니다."
        )

    # -----------------------------------
    # Redundancy planning basis
    # -----------------------------------
    st.markdown("#### Redundancy Planning Basis")

    if "N+1" in redundancy:
        standby_a = 1
        standby_b = 1
        standby_c = 1

    elif "2N" in redundancy:
        standby_a = pod_duty_units
        standby_b = central_duty_units
        standby_c = row_duty_units

    else:
        standby_a = 0
        standby_b = 0
        standby_c = 0

    redundancy_summary = pd.DataFrame(
        [
            {
                "Candidate": "A · Pod-dedicated",
                "Duty Units": pod_duty_units,
                "Standby Basis": standby_a,
                "Installed Units": pod_duty_units + standby_a,
            },
            {
                "Candidate": "B · Central Plant",
                "Duty Units": central_duty_units,
                "Standby Basis": standby_b,
                "Installed Units": central_duty_units + standby_b,
            },
            {
                "Candidate": "C · In-row",
                "Duty Units": row_duty_units,
                "Standby Basis": standby_c,
                "Installed Units": row_count + standby_c,
            },
        ]
    )

    st.dataframe(
        redundancy_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"현재 redundancy 입력조건: {redundancy}. "
        "본 수량은 기본설계 단계의 개념적 설비 구성으로, "
        "실제 CDU redundancy는 배관 격리, 공통부 고장모드, "
        "제어방식 및 유지보수 전략 검토 후 확정합니다."
    )

    # -----------------------------------
    # Selected candidate screening
    # -----------------------------------
    if choice.startswith("A"):
        selected_capacity_ok = pod_capacity_ok

        if pod_multi_cdu_required:
            st.info(
                "선택한 Pod-dedicated 구성에서는 일부 Pod에 "
                "복수 Duty CDU가 필요합니다. 본 단계에서는 동일 Pod 내 "
                "CDU 간 부하 균등분담을 가정합니다."
            )

    elif choice.startswith("B"):
        selected_capacity_ok = True

        st.info(
            "Central CDU Plant는 총 부하 기준 aggregate capacity를 "
            "충족합니다. 실제 적용 전 Hydraulic balancing과 "
            "공통부 redundancy 검토가 필요합니다."
        )

    else:
        selected_capacity_ok = row_capacity_ok

        if row_multi_cdu_required:
            st.info(
                "선택한 In-row 구성에서는 일부 Row Group에 "
                "복수 Duty CDU가 필요합니다. 실제 분기 및 부하분담 방식은 "
                "후속 Hydraulic 검토에서 확인해야 합니다."
            )

    # -----------------------------------
    # Engineer verification
    # -----------------------------------
    check_capacity = st.checkbox(
        "선택 후보의 CDU 용량 및 Loading을 확인했습니다.",
        key="phase2_check_capacity",
    )

    check_redundancy = st.checkbox(
        "Redundancy 구성과 Standby 가정을 확인했습니다.",
        key="phase2_check_redundancy",
    )

    check_layout = st.checkbox(
        "배관 Routing, 유지보수성 및 공간 제약은 후속 상세검토가 필요함을 확인했습니다.",
        key="phase2_check_layout",
    )

    phase2_note = st.text_area(
        "Engineer note",
        key="phase2_note",
        placeholder=(
            "예: Pod 단위 격리를 우선하여 Candidate A를 선정. "
            "Phase 4에서 배관 압력손실 및 CDU 운전조건 추가 검토."
        ),
    )

    ready_phase2 = (
        check_capacity
        and check_redundancy
        and check_layout
        and selected_capacity_ok
    )

    if st.button(
        "✓ Approve Phase 2 Candidate",
        type="primary",
        disabled=not ready_phase2,
        use_container_width=True,
    ):
        st.session_state.topology_choice = choice
        st.session_state.phase2_cdu_capacity = cdu_capacity
        st.session_state.phase2_redundancy = redundancy
        st.session_state.phase2_pods = pods.copy()

        st.session_state.approved[2] = True

        st.success(
            f"Phase 2 approved: {choice}. "
            "승인된 TCS / CDU 설계조건이 다음 Phase로 전달됩니다."
        )

    if st.session_state.approved[2]:
        st.success(
            f"✓ Phase 2 Engineer Review Approved · "
            f"{st.session_state.get('topology_choice', choice)}"
        )

elif phase == 3:
    st.header("Phase 3 · Coolant & Material Validator")

    st.caption(
        "Phase 2에서 승인된 TCS / CDU 구성과 운전온도를 기반으로 "
        "Coolant 물성과 wetted-material 적합성을 검토합니다."
    )

    # -----------------------------------
    # Phase 2 approval gate
    # -----------------------------------
    if not st.session_state.approved[2]:
        st.warning(
            "Phase 2 Engineer Review가 아직 완료되지 않았습니다. "
            "TCS / CDU 후보를 승인한 후 Coolant 검토를 진행해주세요."
        )
        st.stop()

    # -----------------------------------
    # 3A. DESIGN BASIS
    # -----------------------------------
    st.markdown("### 3A · Design Basis")

    b1, b2 = st.columns([1, 4])

    with b1:
        render_tag("PHASE 2 APPROVED", "verified")

    with b2:
        st.write(
            "승인된 TCS / CDU 후보와 TCS Supply / Return Temperature를 "
            "Coolant 검토의 기본 설계조건으로 사용합니다."
        )

    delta_t = return_t - supply_t

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Selected Topology",
        st.session_state.get(
            "topology_choice",
            "Not selected",
        ),
    )

    m2.metric(
        "TCS Supply",
        f"{supply_t:.1f} °C",
    )

    m3.metric(
        "TCS Return",
        f"{return_t:.1f} °C",
    )

    m4.metric(
        "ΔT",
        f"{delta_t:.1f} K",
    )

    if return_t <= supply_t:
        st.error(
            "Return Temperature는 Supply Temperature보다 높아야 합니다."
        )
        st.stop()

    if delta_t < 3:
        st.warning(
            "현재 ΔT가 매우 작습니다. 요구 유량이 증가할 수 있으므로 "
            "Phase 4 Hydraulic 검토에서 유량 및 Pump Power를 확인해야 합니다."
        )

    st.caption(
        "※ 본 온도조건은 설계 입력값이며 특정 OEM의 허용범위를 의미하지 않습니다. "
        "실제 적용 전 장비 제조사의 coolant 및 operating-temperature 요구조건을 확인해야 합니다."
    )
    # -----------------------------------
    # 3B. COOLANT CANDIDATE SCREENING
    # -----------------------------------
    st.divider()

    st.markdown("### 3B · Coolant Candidate Screening")

    c1, c2 = st.columns([1, 4])

    with c1:
        render_tag("SCREENING", "calculated")

    with c2:
        st.write(
            "Coolant 농도를 자동 선정하지 않고, "
            "설계 검토에 사용할 수 있는 후보군의 데이터 확보 수준과 "
            "적용 조건을 비교합니다."
        )

    coolant_candidates = pd.DataFrame(
        [
            {
                "Candidate": "A · Water-based Reference",
                "Purpose": "Baseline comparison",
                "Density kg/m³": "Reference data",
                "Cp kJ/kg·K": "Reference data",
                "Viscosity mPa·s": "Reference data",
                "Approval / Data Status": "Reference only",
            },
            {
                "Candidate": "B · OEM/Vendor-approved Glycol Formulation",
                "Purpose": "Project candidate",
                "Density kg/m³": "Supplier data required",
                "Cp kJ/kg·K": "Supplier data required",
                "Viscosity mPa·s": "Supplier data required",
                "Approval / Data Status": "OEM / supplier confirmation required",
            },
            {
                "Candidate": "C · PG30 Sensitivity Case",
                "Purpose": "Hydraulic sensitivity",
                "Density kg/m³": "1029.85 @ 40°C",
                "Cp kJ/kg·K": "3.841 @ 40°C",
                "Viscosity mPa·s": "1.6295 @ 40°C",
                "Approval / Data Status": "Sensitivity only",
            },
        ]
    )

    st.dataframe(
        coolant_candidates,
        use_container_width=True,
        hide_index=True,
    )
    coolant_a, coolant_b, coolant_c = st.columns(3)

    with coolant_a:
        st.markdown("#### A · Water-based Reference")

        render_tag("REFERENCE", "verified")

        st.write(
            "Water-based coolant를 기준 유체로 사용하여 "
            "유량 및 Hydraulic 결과의 비교 기준을 제공합니다."
        )

        st.info(
            "Reference case이며 실제 프로젝트 적용 승인을 의미하지 않습니다."
        )

    with coolant_b:
        st.markdown("#### B · Approved Formulation")

        render_tag("DATA REQUIRED", "review")

        st.write(
            "OEM 또는 coolant supplier가 허용한 glycol-based formulation을 "
            "실제 프로젝트 후보로 검토합니다."
        )

        st.warning(
            "농도와 물성값은 임의 생성하지 않고 공급사 property table이 필요합니다."
        )

    with coolant_c:
        st.markdown("#### C · PG30 Sensitivity")

        render_tag("SENSITIVITY", "assumption")

        st.write(
            "PG30 예시 물성을 이용해 점도 및 열물성 변화가 "
            "유량과 Pump Power에 미치는 영향을 비교합니다."
        )

        st.warning(
            "본 후보는 OEM 승인 냉각수 선정안이 아니라 민감도 분석용입니다."
        )
    coolant_options = coolant_candidates[
        "Candidate"
    ].tolist()

    coolant_choice = st.radio(
        "Candidate for further engineering review",
        coolant_options,
        horizontal=True,
        key="phase3_coolant_choice",
    )

    if coolant_choice.startswith("A"):
        st.info(
            "Water-based reference가 선택되었습니다. "
            "Phase 4에서는 기준 물성을 사용한 Hydraulic 계산에 활용할 수 있습니다."
        )

    elif coolant_choice.startswith("B"):
        st.warning(
            "실제 적용 후보입니다. Phase 3 승인 전 OEM 또는 공급사의 "
            "물성 데이터와 material compatibility 정보 확인이 필요합니다."
        )

    else:
        st.warning(
            "PG30 sensitivity case가 선택되었습니다. "
            "이 결과는 실제 coolant 선정이 아닌 Hydraulic 영향 비교용으로만 사용합니다."
        )

    st.caption(
        "※ H-LiquidOpt는 coolant 농도를 자율적으로 결정하지 않습니다. "
        "허용 formulation 및 농도범위는 OEM·공급사 요구조건과 "
        "엔지니어 검토를 통해 확정합니다."
    ) 
    st.subheader("Phase 3 · Coolant & Material Candidate Review")
    if not st.session_state.approved[2]:
        st.warning("Phase 2 is pending. Coolant comparison remains exploratory.")
    cdf = load_default_coolants()
    st.dataframe(cdf, use_container_width=True, hide_index=True)
    st.info("Only candidates with explicit numerical property inputs are sent to the hydraulic engine. OEM approval and material compatibility remain validation gates.")
    selectable = cdf[cdf[["rho_kg_m3","cp_kj_kgk","mu_pa_s"]].notna().all(axis=1)]["name"].tolist()
    default = [x for x in ["Water-based reference","PG30 sensitivity fluid"] if x in selectable]
    selected = st.multiselect("Candidates for Phase 4 sensitivity calculation", selectable, default=default)
    st.text_area("Material / OEM review note", key="phase3_note", placeholder="Example: supplier compatibility check required for QD seal.")
    if st.button("✓ Approve Phase 3 candidates", type="primary"):
        if not selected:
            st.error("Select at least one quantitatively defined coolant candidate.")
        else:
            st.session_state.coolant_names = selected
            st.session_state.approved[3] = True
            st.success("Phase 3 candidates saved.")

elif phase == 4:
    st.subheader("Phase 4 · Deterministic Hydraulic Calculation")
    if delta_t <= 0:
        st.error("Return temperature must be greater than supply temperature.")
        st.stop()
    if not st.session_state.approved[3]:
        st.warning("Phase 3 is pending. Default numerical candidates are used for exploratory calculation.")
    cdf = load_default_coolants()
    names = st.session_state.get("coolant_names", ["Water-based reference","PG30 sensitivity fluid"])
    calc_cdf = cdf[cdf["name"].isin(names)].dropna(subset=["rho_kg_m3","cp_kj_kgk","mu_pa_s"]).copy()
    if calc_cdf.empty:
        st.error("No coolant candidate has complete numerical properties.")
        st.stop()
    coolants = [Coolant(row["name"], float(row["rho_kg_m3"]), float(row["cp_kj_kgk"]), float(row["mu_pa_s"]), row.get("status", "")) for _,row in calc_cdf.iterrows()]
    results = evaluate_coolants(st.session_state.racks, coolants, delta_t, geom)

    show_cols = ["coolant","pod","liquid_racks","liquid_load_kw","rack_avg_heat_kw","rack_flow_lpm","pod_flow_lpm","branch_velocity_m_s","network_dp_kpa","rack_dp_kpa","total_dp_kpa","pump_kw"]
    st.dataframe(results[show_cols], use_container_width=True, hide_index=True)
    st.caption("Rack internal ΔP is a synthetic placeholder in this PoC. Replace it with an OEM pressure-flow curve for engineering use.")

    f1,f2 = st.columns(2)
    with f1:
        fig = px.bar(results, x="pod", y="pump_kw", color="coolant", barmode="group", title="Pump electric power by pod")
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        fig2 = px.bar(results, x="pod", y="total_dp_kpa", color="coolant", barmode="group", title="Calculated pressure drop by pod")
        st.plotly_chart(fig2, use_container_width=True)

    st.download_button("Download hydraulic CSV", results.to_csv(index=False).encode("utf-8-sig"), "hliquidopt_hydraulics.csv", "text/csv")
    st.text_area("Engineer calculation review note", key="phase4_note", placeholder="Example: OEM rack pressure-flow curve must replace placeholder before design issue.")
    if st.button("✓ Approve Phase 4 calculation review", type="primary"):
        st.session_state.hydraulic_results = results
        st.session_state.approved[4] = True
        st.success("Phase 4 marked reviewed.")

else:
    st.subheader("Phase 5 · Candidate Comparison & Engineer Decision")
    if delta_t <= 0:
        st.error("Return temperature must be greater than supply temperature.")
        st.stop()
    cdf = load_default_coolants()
    names = st.session_state.get("coolant_names", ["Water-based reference","PG30 sensitivity fluid"])
    calc_cdf = cdf[cdf["name"].isin(names)].dropna(subset=["rho_kg_m3","cp_kj_kgk","mu_pa_s"]).copy()
    coolants = [Coolant(row["name"], float(row["rho_kg_m3"]), float(row["cp_kj_kgk"]), float(row["mu_pa_s"]), row.get("status", "")) for _,row in calc_cdf.iterrows()]
    results = evaluate_coolants(st.session_state.racks, coolants, delta_t, geom)
    ranking = candidate_score_table(results, cdu_capacity)
    pods = pod_summary(st.session_state.racks)

    st.markdown("#### Hydraulic sensitivity ranking")
    st.dataframe(ranking[["rank","coolant","total_pump_kw","worst_dp_kpa","cdu_loading_pct","balanced_score"]], use_container_width=True, hide_index=True)
    best = ranking.iloc[0]
    st.success(
        f"Current balanced hydraulic sensitivity candidate: {best['coolant']} · "
        f"Pump {best['total_pump_kw']:.2f} kW · Worst ΔP {best['worst_dp_kpa']:.1f} kPa"
    )
    st.warning("This ranking is not a final coolant or equipment recommendation. OEM approval, freeze protection, water chemistry, material compatibility and project-specific design criteria remain mandatory.")

    st.markdown("#### Decision history")
    hist = pd.DataFrame([
        ["Phase 1", "Rack / heat-load model", "Approved" if st.session_state.approved[1] else "Pending", st.session_state.get("phase1_note", "")],
        ["Phase 2", st.session_state.get("topology_choice", "Not selected"), "Approved" if st.session_state.approved[2] else "Pending", st.session_state.get("phase2_note", "")],
        ["Phase 3", ", ".join(names), "Approved" if st.session_state.approved[3] else "Pending", st.session_state.get("phase3_note", "")],
        ["Phase 4", "Deterministic hydraulic calculation", "Approved" if st.session_state.approved[4] else "Pending", st.session_state.get("phase4_note", "")],
    ], columns=["Phase","Decision / Result","Status","Engineer note"])
    st.dataframe(hist, use_container_width=True, hide_index=True)

    report = project_report_markdown(st.session_state.racks, pods, ranking, names, delta_t, cdu_capacity, redundancy)
    c1,c2 = st.columns(2)
    c1.download_button("Download design-review report (.md)", report.encode("utf-8-sig"), "H-LiquidOpt_design_review.md", "text/markdown", use_container_width=True)
    c2.download_button("Download current rack dataset (.csv)", st.session_state.racks.to_csv(index=False).encode("utf-8-sig"), "H-LiquidOpt_current_racks.csv", "text/csv", use_container_width=True)

st.divider()
st.caption("Prototype only · Not for construction, procurement, safety certification, or final equipment/coolant selection. Project-specific constraints must be verified by qualified engineers and equipment/coolant suppliers.")
