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
    st.subheader("Phase 1 · Rack Heat Load & Spatial Review")
    st.info("Edit any number of rack rows. HCR converts IT load into liquid-side and residual-air design loads. Layout heat map is shown only when row/col data exist.")
    edited = st.data_editor(
        st.session_state.racks,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "hcr": st.column_config.NumberColumn("HCR", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
            "it_power_kw": st.column_config.NumberColumn("IT Power (kW)", min_value=0.0, step=5.0),
        },
        key="rack_editor",
    )
    if not edited.equals(st.session_state.racks):
        st.session_state.racks = edited
        reset_downstream(1)

    errors = validate_racks(edited)
    if errors:
        st.error("Fix rack data before approval:\n- " + "\n- ".join(errors))
        st.stop()
    calc = heat_loads(edited)
    pods = pod_summary(edited)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Rack positions", len(calc))
    m2.metric("Total IT Load", f"{calc['it_power_kw'].sum()/1000:.2f} MW")
    m3.metric("Liquid Load", f"{calc['liquid_load_kw'].sum()/1000:.2f} MW")
    m4.metric("Residual Air", f"{calc['residual_air_kw'].sum()/1000:.2f} MW")

    left,right = st.columns([1.35,1])
    with left:
        if {"row","col"}.issubset(calc.columns):
            pivot = calc.pivot(index="row", columns="col", values="it_power_kw")
            fig = px.imshow(
                pivot, text_auto=".0f", aspect="auto",
                labels={"color":"IT kW"}, title="Rack Heat-Load Density Map",
                color_continuous_scale="Blues",
            )
            fig.update_xaxes(side="top", title="Column")
            fig.update_yaxes(title="Row", autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No row/col columns: thermal totals still work, but spatial heat map is skipped.")
    with right:
        st.markdown("#### Pod summary")
        st.dataframe(pods, use_container_width=True, hide_index=True)
        hottest = pods.loc[pods["liquid_load_kw"].idxmax()]
        st.markdown(
            f"<div class='phase-card'>Highest liquid-load pod: <b>{hottest['pod']}</b><br>"
            f"{hottest['liquid_load_kw']/1000:.3f} MW liquid load · {int(hottest['liquid_racks'])} liquid-cooled racks</div>",
            unsafe_allow_html=True,
        )
    phase1_note = st.text_area("Engineer note", key="phase1_note", placeholder="Check OEM HCR, rack power and pod assignment.")
    if st.button("✓ Approve Phase 1", type="primary"):
        st.session_state.approved[1] = True
        st.success("Phase 1 approved. Downstream results can now use this rack dataset.")

elif phase == 2:
    st.subheader("Phase 2 · TCS / CDU Candidate Review")
    if not st.session_state.approved[1]:
        st.warning("Phase 1 is pending. Results below are exploratory until the engineer approves the load model.")
    pods = pod_summary(st.session_state.racks)
    pods["CDU loading %"] = pods["liquid_load_kw"] / (cdu_capacity*1000) * 100
    total_liquid = pods["liquid_load_kw"].sum()
    min_duty = recommended_duty_cdus(total_liquid, cdu_capacity)

    m1,m2,m3 = st.columns(3)
    m1.metric("Total liquid load", f"{total_liquid/1000:.2f} MW")
    m2.metric("Minimum aggregate duty CDU", f"{min_duty} unit(s)")
    m3.metric("Max pod loading", f"{pods['CDU loading %'].max():.1f}%")
    st.dataframe(pods, use_container_width=True, hide_index=True)

    options = pd.DataFrame([
        ["A", "One CDU per Pod + shared standby", "Clear pod boundary / simple isolation", "More CDU units; verify each pod fits one duty CDU"],
        ["B", "Central CDU plant + branch headers", "Central maintenance / capacity pooling", "Longer network and balancing complexity"],
        ["C", "In-row CDU grouping", "Short secondary loop / close to load", "White-space footprint and service access"],
    ], columns=["Option","Topology","Strength","Engineer review point"])
    st.markdown("#### Candidate topologies")
    st.dataframe(options, use_container_width=True, hide_index=True)
    choice = st.selectbox("Engineer-selected topology", options["Topology"].tolist(), index=0)
    st.text_area("Engineer note", key="phase2_note", placeholder="Example: prioritize pod isolation and maintenance access.")
    if (pods["CDU loading %"] > 100).any() and choice.startswith("One CDU per Pod"):
        st.error("At least one pod exceeds one candidate CDU's nominal capacity. Increase capacity, split the pod, or choose another topology.")
    if st.button("✓ Approve Phase 2 candidate", type="primary"):
        st.session_state.topology_choice = choice
        st.session_state.approved[2] = True
        st.success("Phase 2 candidate saved.")

elif phase == 3:
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
