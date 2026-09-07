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

from ai_adapter import (
    test_openai_connection,
    extract_specification,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(
    page_title="H-LiquidOpt",
    page_icon="💧",
    layout="wide",
)

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
# =========================================
# Protect persistent project state
# from Streamlit widget cleanup
# =========================================
PERSISTENT_PROJECT_KEYS = [
    "supply_t",
    "return_t",
    "cdu_capacity",
    "redundancy",
    "common_d",
    "row_d",
    "branch_d",
    "common_l",
    "row_l",
    "branch_l",
    "rack_dp",
]

for key in PERSISTENT_PROJECT_KEYS:
    if key in st.session_state:
        # Re-assigning the key interrupts Streamlit's
        # widget cleanup behavior and keeps it as
        # persistent project state.
        st.session_state[key] = st.session_state[key]
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
# =========================================
# AI SPECIFICATION ASSISTANT
# =========================================
with st.expander("🤖 AI Specification Assistant", expanded=False):

    st.caption(
        "Upload an OEM datasheet, rack specification, CDU datasheet, "
        "or coolant datasheet. AI extracts candidate engineering inputs "
        "for engineer verification."
    )

    uploaded_spec = st.file_uploader(
        "Upload specification PDF",
        type=["pdf"],
        key="ai_spec_pdf",
    )

    authorized = st.checkbox(
        "I confirm that I am authorized to process this document.",
        key="ai_spec_authorized",
    )

    if uploaded_spec is not None:
        st.write(
            f"**Selected file:** {uploaded_spec.name}"
        )

        analyze_disabled = not authorized

        if st.button(
            "Analyze Specification",
            type="primary",
            disabled=analyze_disabled,
            key="analyze_specification_button",
        ):
            try:
                api_key = st.secrets.get(
                    "OPENAI_API_KEY"
                )

                if not api_key:
                    st.error(
                        "OPENAI_API_KEY was not found in Streamlit Secrets."
                    )

                else:
                    with st.spinner(
                        "AI is extracting engineering specification candidates..."
                    ):
                        result = extract_specification(
                            file_bytes=uploaded_spec.getvalue(),
                            filename=uploaded_spec.name,
                            api_key=api_key,
                    )
                    # Clear previous engineer-edit widget states
                    for key in list(st.session_state.keys()):
                        if key.startswith("ai_edit_") or key.startswith("ai_verify_"):
                            st.session_state.pop(key, None)

                    st.session_state.pop("ai_verified_spec", None)                    
                    st.session_state["ai_spec_result"] = result

                    st.success(
                        "Specification extraction completed. "
                        "Review all values before using them."
                    )

            except Exception as e:
                st.error(
                    f"Specification extraction failed: {e}"
                )

    if "ai_spec_result" in st.session_state:

        result = st.session_state["ai_spec_result"]
    
        # =========================================
        # EXTRACTED RESULT
        # =========================================
        st.markdown(
            "#### Extracted Specification Candidates"
        )
    
        summary_rows = [
            ["Document Type", result.get("document_type"), ""],
            ["Manufacturer", result.get("manufacturer"), ""],
            ["Model", result.get("model"), ""],
            ["Rated Power", result.get("rated_power_kw"), "kW"],
            ["HCR", result.get("hcr"), ""],
            ["Coolant", result.get("coolant_name"), ""],
            ["Density", result.get("density_kg_m3"), "kg/m³"],
            ["Specific Heat", result.get("cp_kj_kgk"), "kJ/kg·K"],
            ["Viscosity", result.get("viscosity_mpas"), "mPa·s"],
            ["Property Temperature", result.get("property_temp_c"), "°C"],
            ["Supply Temp Min", result.get("supply_temp_min_c"), "°C"],
            ["Supply Temp Max", result.get("supply_temp_max_c"), "°C"],
            ["Recommended Flow", result.get("recommended_flow_lpm"), "L/min"],
            ["Pressure Drop", result.get("pressure_drop_kpa"), "kPa"],
        ]
    
        normalized_rows = []
    
        for field, value, unit in summary_rows:
            normalized_rows.append(
                {
                    "Field": field,
                    "Extracted Value": (
                        value
                        if value is not None
                        else "Not found"
                    ),
                    "Unit": unit,
                }
            )
    
        st.dataframe(
            pd.DataFrame(normalized_rows),
            use_container_width=True,
            hide_index=True,
        )
    
        # =========================================
        # SOURCE TRACEABILITY
        # =========================================
        sources = result.get(
            "sources",
            [],
        )
    
        if sources:
            st.markdown(
                "#### Source Traceability"
            )
    
            st.dataframe(
                pd.DataFrame(sources),
                use_container_width=True,
                hide_index=True,
            )
    
        notes = result.get(
            "notes",
            [],
        )
    
        if notes:
            st.markdown(
                "#### AI Review Notes"
            )
    
            for note in notes:
                st.write(
                    f"- {note}"
                )
    
        st.warning(
            "AI-extracted values are candidate inputs only. "
            "They are not transferred to the deterministic workflow "
            "until an engineer reviews and saves them."
        )
    
        # =========================================
        # ENGINEER VERIFICATION
        # =========================================
        st.divider()
    
        st.markdown(
            "#### Engineer Verification"
        )
    
        st.caption(
            "Review the source evidence above, correct any AI-extracted "
            "value if necessary, and verify each applicable group."
        )
    
        def display_value(value):
            if value is None:
                return ""
            return str(value)
    
        def parse_optional_float(value, field_name):
            value = value.strip()
    
            if value == "":
                return None
    
            try:
                return float(value)
    
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} must be numeric or blank."
                ) from exc
    
        # -----------------------------------------
        # Equipment / Rack Identity
        # -----------------------------------------
        st.markdown(
            "##### A · Equipment / Rack"
        )
    
        e1, e2, e3 = st.columns(3)
    
        with e1:
            engineer_manufacturer = st.text_input(
                "Manufacturer",
                value=display_value(
                    result.get("manufacturer")
                ),
                key="ai_edit_manufacturer",
            )
    
        with e2:
            engineer_model = st.text_input(
                "Model",
                value=display_value(
                    result.get("model")
                ),
                key="ai_edit_model",
            )
    
        with e3:
            engineer_document_type = st.text_input(
                "Document Type",
                value=display_value(
                    result.get("document_type")
                ),
                key="ai_edit_document_type",
            )
    
        e4, e5 = st.columns(2)
    
        with e4:
            engineer_power = st.text_input(
                "Rated Power · kW",
                value=display_value(
                    result.get("rated_power_kw")
                ),
                key="ai_edit_rated_power_kw",
            )
    
        with e5:
            engineer_hcr = st.text_input(
                "HCR · 0–1",
                value=display_value(
                    result.get("hcr")
                ),
                key="ai_edit_hcr",
            )
    
        rack_group_present = any(
            result.get(field) is not None
            for field in [
                "manufacturer",
                "model",
                "rated_power_kw",
                "hcr",
            ]
        )
    
        if rack_group_present:
            verify_rack = st.checkbox(
                "I verified the equipment identity, rated power and HCR against the source document.",
                key="ai_verify_rack",
            )
        else:
            verify_rack = True
    
        # -----------------------------------------
        # Coolant Properties
        # -----------------------------------------
        st.markdown(
            "##### B · Coolant Properties"
        )
    
        c1, c2 = st.columns(2)
    
        with c1:
            engineer_coolant = st.text_input(
                "Coolant / Formulation",
                value=display_value(
                    result.get("coolant_name")
                ),
                key="ai_edit_coolant_name",
            )
    
            engineer_density = st.text_input(
                "Density · kg/m³",
                value=display_value(
                    result.get("density_kg_m3")
                ),
                key="ai_edit_density",
            )
    
            engineer_cp = st.text_input(
                "Specific Heat · kJ/kg·K",
                value=display_value(
                    result.get("cp_kj_kgk")
                ),
                key="ai_edit_cp",
            )
    
        with c2:
            engineer_viscosity = st.text_input(
                "Dynamic Viscosity · mPa·s",
                value=display_value(
                    result.get("viscosity_mpas")
                ),
                key="ai_edit_viscosity",
            )
    
            engineer_property_temp = st.text_input(
                "Property Reference Temperature · °C",
                value=display_value(
                    result.get("property_temp_c")
                ),
                key="ai_edit_property_temp",
            )
    
        coolant_group_present = any(
            result.get(field) is not None
            for field in [
                "coolant_name",
                "density_kg_m3",
                "cp_kj_kgk",
                "viscosity_mpas",
                "property_temp_c",
            ]
        )
    
        if coolant_group_present:
            verify_coolant = st.checkbox(
                "I verified the coolant identity and thermophysical properties against the source document.",
                key="ai_verify_coolant",
            )
        else:
            verify_coolant = True
    
        # -----------------------------------------
        # Hydraulic / Operating Constraints
        # -----------------------------------------
        st.markdown(
            "##### C · Operating / Hydraulic Data"
        )
    
        h1, h2 = st.columns(2)
    
        with h1:
            engineer_supply_min = st.text_input(
                "Supply Temperature Min · °C",
                value=display_value(
                    result.get("supply_temp_min_c")
                ),
                key="ai_edit_supply_min",
            )
    
            engineer_supply_max = st.text_input(
                "Supply Temperature Max · °C",
                value=display_value(
                    result.get("supply_temp_max_c")
                ),
                key="ai_edit_supply_max",
            )
    
        with h2:
            engineer_flow = st.text_input(
                "Recommended Flow · L/min",
                value=display_value(
                    result.get("recommended_flow_lpm")
                ),
                key="ai_edit_flow",
            )
    
            engineer_dp = st.text_input(
                "Pressure Drop · kPa",
                value=display_value(
                    result.get("pressure_drop_kpa")
                ),
                key="ai_edit_dp",
            )
    
        hydraulic_group_present = any(
            result.get(field) is not None
            for field in [
                "supply_temp_min_c",
                "supply_temp_max_c",
                "recommended_flow_lpm",
                "pressure_drop_kpa",
            ]
        )
    
        if hydraulic_group_present:
            verify_hydraulic = st.checkbox(
                "I verified the operating temperature, flow and pressure-drop conditions against the source document.",
                key="ai_verify_hydraulic",
            )
        else:
            verify_hydraulic = True
    
        # =========================================
        # SAVE VERIFIED SNAPSHOT
        # =========================================
        verification_ready = (
            verify_rack
            and verify_coolant
            and verify_hydraulic
        )
    
        if st.button(
            "✓ Save Engineer-Verified Specification",
            type="primary",
            disabled=not verification_ready,
            key="save_ai_verified_spec",
            use_container_width=True,
        ):
            try:
                verified_power = parse_optional_float(
                    engineer_power,
                    "Rated Power",
                )
    
                verified_hcr = parse_optional_float(
                    engineer_hcr,
                    "HCR",
                )
    
                verified_density = parse_optional_float(
                    engineer_density,
                    "Density",
                )
    
                verified_cp = parse_optional_float(
                    engineer_cp,
                    "Specific Heat",
                )
    
                verified_viscosity = parse_optional_float(
                    engineer_viscosity,
                    "Dynamic Viscosity",
                )
    
                verified_property_temp = parse_optional_float(
                    engineer_property_temp,
                    "Property Reference Temperature",
                )
    
                verified_supply_min = parse_optional_float(
                    engineer_supply_min,
                    "Supply Temperature Min",
                )
    
                verified_supply_max = parse_optional_float(
                    engineer_supply_max,
                    "Supply Temperature Max",
                )
    
                verified_flow = parse_optional_float(
                    engineer_flow,
                    "Recommended Flow",
                )
    
                verified_dp = parse_optional_float(
                    engineer_dp,
                    "Pressure Drop",
                )
    
                # Basic engineering sanity checks
                if (
                    verified_power is not None
                    and verified_power <= 0
                ):
                    raise ValueError(
                        "Rated Power must be greater than 0 kW."
                    )
    
                if (
                    verified_hcr is not None
                    and not 0 <= verified_hcr <= 1
                ):
                    raise ValueError(
                        "HCR must be between 0 and 1."
                    )
    
                for field_name, field_value in [
                    ("Density", verified_density),
                    ("Specific Heat", verified_cp),
                    ("Dynamic Viscosity", verified_viscosity),
                    ("Recommended Flow", verified_flow),
                    ("Pressure Drop", verified_dp),
                ]:
                    if (
                        field_value is not None
                        and field_value <= 0
                    ):
                        raise ValueError(
                            f"{field_name} must be greater than 0."
                        )
    
                if (
                    verified_supply_min is not None
                    and verified_supply_max is not None
                    and verified_supply_min > verified_supply_max
                ):
                    raise ValueError(
                        "Supply Temperature Min cannot be greater than Supply Temperature Max."
                    )
    
                st.session_state["ai_verified_spec"] = {
                    "document_type": (
                        engineer_document_type.strip()
                        or None
                    ),
                    "manufacturer": (
                        engineer_manufacturer.strip()
                        or None
                    ),
                    "model": (
                        engineer_model.strip()
                        or None
                    ),
                    "rated_power_kw": verified_power,
                    "hcr": verified_hcr,
                    "coolant_name": (
                        engineer_coolant.strip()
                        or None
                    ),
                    "density_kg_m3": verified_density,
                    "cp_kj_kgk": verified_cp,
                    "viscosity_mpas": verified_viscosity,
                    "property_temp_c": verified_property_temp,
                    "supply_temp_min_c": verified_supply_min,
                    "supply_temp_max_c": verified_supply_max,
                    "recommended_flow_lpm": verified_flow,
                    "pressure_drop_kpa": verified_dp,
                    "sources": result.get(
                        "sources",
                        [],
                    ),
                    "ai_notes": result.get(
                        "notes",
                        [],
                    ),
                    "verified_at": datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
    
                st.success(
                    "Engineer-verified specification saved. "
                    "Only this verified snapshot will be eligible "
                    "for transfer to H-LiquidOpt."
                )
    
            except ValueError as e:
                st.error(str(e))
    
        # =========================================
        # VERIFIED SNAPSHOT STATUS
        # =========================================
        if "ai_verified_spec" in st.session_state:
            verified = st.session_state[
                "ai_verified_spec"
            ]
    
            st.success(
                "✓ ENGINEER VERIFIED · "
                f"{verified.get('manufacturer') or 'Unknown manufacturer'} "
                f"{verified.get('model') or ''}"
            )
    
            st.caption(
                f"Verified snapshot saved at "
                f"{verified.get('verified_at', '-')}. "
                "The original AI extraction remains separate "
                "from the engineer-approved data."
            )
            # =========================================
            # TRANSFER VERIFIED VALUES
            # =========================================
            st.markdown(
                "#### Transfer Verified Values"
            )
    
            st.caption(
                "Only engineer-verified values can be transferred. "
                "Transferred values remain editable in each design phase."
            )
    
            t1, t2, t3 = st.columns(3)
    
            # -----------------------------------------
            # Phase 1 transfer staging
            # -----------------------------------------
            with t1:
                phase1_transfer_ready = (
                    verified.get("rated_power_kw") is not None
                    and verified.get("hcr") is not None
                )
    
                if st.button(
                    "→ Send to Phase 1",
                    disabled=not phase1_transfer_ready,
                    key="transfer_ai_to_phase1",
                    use_container_width=True,
                ):
                    st.session_state["ai_phase1_import_pending"] = {
                        "manufacturer": verified.get("manufacturer"),
                        "model": verified.get("model"),
                        "rated_power_kw": verified.get("rated_power_kw"),
                        "hcr": verified.get("hcr"),
                    }
    
                    st.success(
                        "Verified rack/equipment values staged for Phase 1."
                    )
    
            # -----------------------------------------
            # Phase 3 transfer staging
            # -----------------------------------------
            with t2:
                phase3_transfer_ready = (
                    verified.get("coolant_name") is not None
                    and verified.get("density_kg_m3") is not None
                    and verified.get("cp_kj_kgk") is not None
                    and verified.get("viscosity_mpas") is not None
                )
    
                if st.button(
                    "→ Send to Phase 3",
                    disabled=not phase3_transfer_ready,
                    key="transfer_ai_to_phase3",
                    use_container_width=True,
                ):
                    st.session_state["ai_phase3_import_pending"] = {
                        "name": verified.get("coolant_name"),
                        "rho_kg_m3": verified.get("density_kg_m3"),
                        "cp_kj_kgk": verified.get("cp_kj_kgk"),
                        "mu_pa_s": (
                            verified.get("viscosity_mpas") / 1000
                            if verified.get("viscosity_mpas") is not None
                            else None
                        ),
                        "viscosity_mpas": verified.get("viscosity_mpas"),
                        "property_temp_c": verified.get("property_temp_c"),
                        "source": (
                            f"AI-extracted and engineer-verified · "
                            f"{verified.get('manufacturer') or ''} "
                            f"{verified.get('model') or ''}"
                        ).strip(),
                    }
    
                    st.success(
                        "Verified coolant properties staged for Phase 3."
                    )
    
            # -----------------------------------------
            # Phase 4 reference transfer staging
            # -----------------------------------------
            with t3:
                phase4_transfer_ready = any(
                    verified.get(field) is not None
                    for field in [
                        "supply_temp_min_c",
                        "supply_temp_max_c",
                        "recommended_flow_lpm",
                        "pressure_drop_kpa",
                    ]
                )
    
                if st.button(
                    "→ Send to Phase 4",
                    disabled=not phase4_transfer_ready,
                    key="transfer_ai_to_phase4",
                    use_container_width=True,
                ):
                    st.session_state["ai_phase4_reference_pending"] = {
                        "supply_temp_min_c": verified.get(
                            "supply_temp_min_c"
                        ),
                        "supply_temp_max_c": verified.get(
                            "supply_temp_max_c"
                        ),
                        "recommended_flow_lpm": verified.get(
                            "recommended_flow_lpm"
                        ),
                        "pressure_drop_kpa": verified.get(
                            "pressure_drop_kpa"
                        ),
                    }
    
                    st.success(
                        "Verified OEM operating data staged for Phase 4 review."
                    )
    st.divider()

    if st.button(
        "Test OpenAI Connection",
        key="test_openai_connection_button",
    ):
        try:
            api_key = st.secrets.get(
                "OPENAI_API_KEY"
            )

            result = test_openai_connection(
                api_key=api_key
            )

            st.success(result)

        except Exception as e:
            st.error(
                f"OpenAI connection failed: {e}"
            )
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
# =========================================
# Persistent Sidebar State Helpers
# =========================================
def restore_sidebar_widget(
    widget_key,
    project_key,
    default_value,
):
    """
    Restore a temporary widget value from
    the persistent project state.
    """
    if widget_key not in st.session_state:
        st.session_state[widget_key] = (
            st.session_state.get(
                project_key,
                default_value,
            )
        )


def sync_sidebar_widget(
    widget_key,
    project_key,
    from_phase,
):
    """
    Save a temporary widget value into the
    persistent project state.

    If an engineering input changes,
    downstream approvals are reset.
    """
    old_value = st.session_state.get(
        project_key
    )

    new_value = st.session_state[
        widget_key
    ]

    st.session_state[
        project_key
    ] = new_value

    if old_value != new_value:
        reset_downstream(
            from_phase
        )
with st.sidebar:
    st.header("H-LiquidOpt")
    st.caption(
        f"Design workflow · Phase {phase} of 5"
    )

    # =========================================
    # PHASE 1
    # =========================================
    if phase == 1:
        st.subheader("Rack / IT Inputs")

        st.caption(
            "Upload the project rack dataset "
            "or use the built-in demo case."
        )

        template = pd.DataFrame(
            {
                "rack_id": ["R001", "R002"],
                "pod": ["A", "A"],
                "rack_type": [
                    "Compute",
                    "Support",
                ],
                "it_power_kw": [
                    120.0,
                    20.0,
                ],
                "hcr": [
                    0.85,
                    0.0,
                ],
                "row": [1, 1],
                "col": [1, 2],
            }
        )

        st.download_button(
            "Download Rack CSV template",
            template.to_csv(
                index=False
            ).encode(
                "utf-8-sig"
            ),
            "hliquidopt_rack_template.csv",
            "text/csv",
            use_container_width=True,
        )

        uploaded = st.file_uploader(
            "Upload Rack CSV",
            type=["csv"],
            key="phase1_rack_csv_upload",
        )

        if uploaded is not None:
            new_racks = pd.read_csv(
                uploaded
            )

            # Keep the uploaded file identity
            st.session_state[
                "phase1_dataset_name"
            ] = uploaded.name

            if not new_racks.equals(
                st.session_state.racks
            ):
                st.session_state.racks = (
                    new_racks
                )

                reset_downstream(1)

        if st.button(
            "Restore 48-rack demo",
            key="phase1_restore_demo",
            use_container_width=True,
        ):
            st.session_state.racks = (
                load_default_racks()
            )

            st.session_state[
                "phase1_dataset_name"
            ] = "Built-in 48-rack demo"

            reset_downstream(1)

            st.rerun()

        current_dataset_name = (
            st.session_state.get(
                "phase1_dataset_name",
                "Built-in 48-rack demo",
            )
        )

        st.info(
            f"Current dataset · "
            f"{current_dataset_name}\n\n"
            f"{len(st.session_state.racks)} "
            f"rack records loaded"
        )

        st.divider()

        st.caption(
            "The file uploader itself may appear empty "
            "after leaving this phase, but the loaded "
            "rack dataset remains stored in the project session."
        )

    # =========================================
    # PHASE 2
    # =========================================
    elif phase == 2:
        st.subheader("TCS / CDU Inputs")

        # Restore temporary widget state
        restore_sidebar_widget(
            "_phase2_cdu_capacity",
            "cdu_capacity",
            2.0,
        )

        restore_sidebar_widget(
            "_phase2_redundancy",
            "redundancy",
            "N+1 shared standby",
        )

        st.number_input(
            "CDU Candidate Capacity (MW/unit)",
            min_value=0.1,
            step=0.1,
            key="_phase2_cdu_capacity",
            on_change=sync_sidebar_widget,
            args=(
                "_phase2_cdu_capacity",
                "cdu_capacity",
                2,
            ),
        )

        redundancy_options = [
            "N+1 shared standby",
            "N",
            "2N",
        ]

        # Safety check for an old/invalid state
        if (
            st.session_state[
                "_phase2_redundancy"
            ]
            not in redundancy_options
        ):
            st.session_state[
                "_phase2_redundancy"
            ] = "N+1 shared standby"

            st.session_state[
                "redundancy"
            ] = "N+1 shared standby"

        st.selectbox(
            "Redundancy",
            redundancy_options,
            key="_phase2_redundancy",
            on_change=sync_sidebar_widget,
            args=(
                "_phase2_redundancy",
                "redundancy",
                2,
            ),
        )

        st.divider()

        st.caption(
            "These values are used to screen "
            "TCS/CDU topology candidates."
        )

    # =========================================
    # PHASE 3
    # =========================================
    elif phase == 3:
        st.subheader("Coolant Conditions")

        restore_sidebar_widget(
            "_phase3_supply_t",
            "supply_t",
            35.0,
        )

        restore_sidebar_widget(
            "_phase3_return_t",
            "return_t",
            45.0,
        )

        st.number_input(
            "TCS Supply Temperature (°C)",
            step=1.0,
            key="_phase3_supply_t",
            on_change=sync_sidebar_widget,
            args=(
                "_phase3_supply_t",
                "supply_t",
                3,
            ),
        )

        st.number_input(
            "TCS Return Temperature (°C)",
            step=1.0,
            key="_phase3_return_t",
            on_change=sync_sidebar_widget,
            args=(
                "_phase3_return_t",
                "return_t",
                3,
            ),
        )

        current_dt = (
            float(
                st.session_state.get(
                    "return_t",
                    45.0,
                )
            )
            - float(
                st.session_state.get(
                    "supply_t",
                    35.0,
                )
            )
        )

        st.metric(
            "Design ΔT",
            f"{current_dt:.1f} K",
        )

        if current_dt <= 0:
            st.error(
                "Return temperature must be "
                "greater than supply temperature."
            )

        st.divider()

        st.caption(
            "Coolant selection remains subject "
            "to OEM and supplier validation."
        )

    # =========================================
    # PHASE 4
    # =========================================
    elif phase == 4:
        st.subheader("Hydraulic Inputs")

        # -------------------------------------
        # Restore persistent values
        # -------------------------------------
        restore_sidebar_widget(
            "_phase4_common_d",
            "common_d",
            0.2027,
        )

        restore_sidebar_widget(
            "_phase4_row_d",
            "row_d",
            0.1541,
        )

        restore_sidebar_widget(
            "_phase4_branch_d",
            "branch_d",
            0.0525,
        )

        restore_sidebar_widget(
            "_phase4_common_l",
            "common_l",
            20.0,
        )

        restore_sidebar_widget(
            "_phase4_row_l",
            "row_l",
            12.0,
        )

        restore_sidebar_widget(
            "_phase4_branch_l",
            "branch_l",
            6.0,
        )

        restore_sidebar_widget(
            "_phase4_rack_dp",
            "rack_dp",
            120.0,
        )

        # -------------------------------------
        # Pipe geometry
        # -------------------------------------
        st.markdown(
            "**Pipe geometry**"
        )

        st.number_input(
            "Common pipe ID (m)",
            min_value=0.001,
            format="%.4f",
            key="_phase4_common_d",
            on_change=sync_sidebar_widget,
            args=(
                "_phase4_common_d",
                "common_d",
                4,
            ),
        )

        st.number_input(
            "Row header ID (m)",
            min_value=0.001,
            format="%.4f",
            key="_phase4_row_d",
            on_change=sync_sidebar_widget,
            args=(
                "_phase4_row_d",
                "row_d",
                4,
            ),
        )

        st.number_input(
            "Rack branch ID (m)",
            min_value=0.001,
            format="%.4f",
            key="_phase4_branch_d",
            on_change=sync_sidebar_widget,
            args=(
                "_phase4_branch_d",
                "branch_d",
                4,
            ),
        )

        # -------------------------------------
        # Equivalent lengths
        # -------------------------------------
        st.markdown(
            "**Equivalent supply + return length**"
        )

        st.number_input(
            "Common pipe length (m)",
            min_value=0.0,
            step=1.0,
            key="_phase4_common_l",
            on_change=sync_sidebar_widget,
            args=(
                "_phase4_common_l",
                "common_l",
                4,
            ),
        )

        st.number_input(
            "Row header length (m)",
            min_value=0.0,
            step=1.0,
            key="_phase4_row_l",
            on_change=sync_sidebar_widget,
            args=(
                "_phase4_row_l",
                "row_l",
                4,
            ),
        )

        st.number_input(
            "Rack branch length (m)",
            min_value=0.0,
            step=1.0,
            key="_phase4_branch_l",
            on_change=sync_sidebar_widget,
            args=(
                "_phase4_branch_l",
                "branch_l",
                4,
            ),
        )

        st.divider()

        # -------------------------------------
        # Rack pressure-drop assumption
        # -------------------------------------
        st.number_input(
            "Synthetic rack ΔP reference (kPa)",
            min_value=0.0,
            step=5.0,
            key="_phase4_rack_dp",
            on_change=sync_sidebar_widget,
            args=(
                "_phase4_rack_dp",
                "rack_dp",
                4,
            ),
            help=(
                "PoC placeholder only. "
                "Final engineering use requires "
                "an OEM pressure-flow curve."
            ),
        )

        st.warning(
            "Rack ΔP is currently an ASSUMPTION "
            "used for prototype sensitivity."
        )

    # =========================================
    # PHASE 5
    # =========================================
    else:
        st.subheader("Final Review")

        st.caption(
            "No new engineering inputs "
            "are required in this phase."
        )

        approved_count = sum(
            bool(
                st.session_state.approved[p]
            )
            for p in [
                1,
                2,
                3,
                4,
            ]
        )

        st.metric(
            "Approved phases",
            f"{approved_count} / 4",
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
        
        # ===================================
        # IMPORT ENGINEER-VERIFIED AI DATA
        # ===================================
        if "ai_phase1_import_pending" in st.session_state:
            ai_import = st.session_state.pop(
                "ai_phase1_import_pending"
            )

            manufacturer = (
                ai_import.get("manufacturer")
                or ""
            )

            model = (
                ai_import.get("model")
                or "AI Imported Equipment"
            )

            equipment_label = (
                f"{manufacturer} {model}"
            ).strip()

            ai_equipment_row = {
                "rack_id": "R01",
                "equipment": equipment_label,
                "quantity": 1,
                "rated_power_kw": float(
                    ai_import.get("rated_power_kw")
                    or 0.0
                ),
                "load_factor": 1.00,
                "hcr": float(
                    ai_import.get("hcr")
                    or 0.0
                ),
                "pod": "A",
                "row": 1,
                "col": 1,
            }

            st.session_state.equipment_input = pd.DataFrame(
                [ai_equipment_row]
            )

            st.session_state["phase1_ai_imported"] = True
        if st.session_state.get(
            "phase1_ai_imported",
            False,
        ):
            st.success(
                "✓ Engineer-verified AI equipment specification loaded. "
                "Review the equipment configuration before Phase 1 approval."
            )
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
    st.header("Phase 3 · Coolant Property & Compatibility Review")

    st.caption(
        "Phase 2에서 승인된 설계조건을 기반으로 Baseline, Project Candidate, "
        "Sensitivity Case의 물성을 비교하고 Phase 4 Hydraulic 계산에 사용할 데이터를 검증합니다."
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

    if "phase1_racks" not in st.session_state:
        st.error(
            "승인된 Phase 1 Rack Load Model이 없습니다. "
            "Phase 1부터 다시 승인해주세요."
        )
        st.stop()

    # -----------------------------------
    # Load reference coolant database
    # -----------------------------------
    cdf = load_default_coolants()

    phase3_racks = st.session_state.phase1_racks.copy()
    phase3_calc = heat_loads(phase3_racks)

    total_liquid_kw = phase3_calc["liquid_load_kw"].sum()

    delta_t = return_t - supply_t
    mean_fluid_temp = (supply_t + return_t) / 2.0

    if delta_t <= 0:
        st.error(
            "Return Temperature는 Supply Temperature보다 높아야 합니다."
        )
        st.stop()

    # ===================================
    # 3A · DESIGN BASIS
    # ===================================
    st.markdown("### 3A · Design Basis")

    a1, a2 = st.columns([1, 4])

    with a1:
        render_tag("PHASE 2 APPROVED", "verified")

    with a2:
        st.write(
            "승인된 Liquid Heat Load와 TCS Supply / Return Temperature를 "
            "Coolant 후보 비교의 공통 설계조건으로 사용합니다."
        )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Liquid Heat Load",
        f"{total_liquid_kw / 1000:.2f} MW",
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
        "Design ΔT",
        f"{delta_t:.1f} K",
    )

    st.caption(
        f"Property comparison target temperature: 약 {mean_fluid_temp:.1f} °C "
        "(Supply / Return 평균온도 기준). "
        "본 온도는 특정 OEM의 허용조건을 의미하지 않습니다."
    )

    # ===================================
    # 3B · BASELINE & CANDIDATE DEFINITION
    # ===================================
    st.divider()

    st.markdown("### 3B · Baseline & Candidate Definition")

    b1, b2 = st.columns([1, 4])

    with b1:
        render_tag("INPUT / REFERENCE", "input")

    with b2:
        st.write(
            "A는 비교 기준, B는 실제 프로젝트 검토 후보, "
            "C는 Hydraulic sensitivity 분석용 후보입니다."
        )

    # -----------------------------------
    # A · Baseline Reference
    # -----------------------------------
    st.markdown("#### A · Baseline Reference")

    baseline_mode = st.radio(
        "Baseline mode",
        [
            "Default Pure-water Reference",
            "Custom Project Baseline",
        ],
        horizontal=True,
        key="phase3_baseline_mode",
    )

    water_match = cdf[
        cdf["name"] == "Water-based reference"
    ]

    if water_match.empty:
        st.error(
            "coolants.csv에서 Water-based reference를 찾을 수 없습니다."
        )
        st.stop()

    water_row = water_match.iloc[0]

    if baseline_mode == "Default Pure-water Reference":

        baseline_name = "Pure-water Reference"

        baseline_rho = float(
            water_row["rho_kg_m3"]
        )

        baseline_cp = float(
            water_row["cp_kj_kgk"]
        )

        baseline_mu_pa_s = float(
            water_row["mu_pa_s"]
        )

        if (
            "property_temp_c" in water_row.index
            and pd.notna(water_row["property_temp_c"])
        ):
            baseline_temp = float(
                water_row["property_temp_c"]
            )
        else:
            baseline_temp = 40.0

        if (
            "source" in water_row.index
            and pd.notna(water_row["source"])
        ):
            baseline_source = str(
                water_row["source"]
            )
        else:
            baseline_source = (
                "Pure-water thermophysical reference dataset"
            )

        baseline_data_ready = True

        render_tag("REFERENCE", "verified")

        st.info(
            "기본 A는 약 40°C 순수 물의 thermophysical property를 사용하는 "
            "비교 기준입니다. 실제 D2C loop coolant 승인안을 의미하지 않습니다."
        )

    else:
        render_tag("USER INPUT", "input")

        baseline_name = st.text_input(
            "Baseline Fluid Name",
            value="Existing Project Baseline",
            key="phase3_baseline_name",
        )

        cA1, cA2, cA3 = st.columns(3)

        with cA1:
            baseline_rho = st.number_input(
                "Baseline Density (kg/m³)",
                min_value=0.0,
                value=992.2,
                step=1.0,
                key="phase3_baseline_rho",
            )

        with cA2:
            baseline_cp = st.number_input(
                "Baseline Cp (kJ/kg·K)",
                min_value=0.0,
                value=4.179,
                step=0.01,
                key="phase3_baseline_cp",
            )

        with cA3:
            baseline_mu_mpas = st.number_input(
                "Baseline Viscosity (mPa·s)",
                min_value=0.0,
                value=0.653,
                step=0.01,
                key="phase3_baseline_mu",
            )

        baseline_mu_pa_s = (
            baseline_mu_mpas / 1000.0
        )

        baseline_temp = st.number_input(
            "Baseline Property Temperature (°C)",
            value=float(mean_fluid_temp),
            step=1.0,
            key="phase3_baseline_temp",
        )

        baseline_source = st.text_input(
            "Baseline Data Source",
            placeholder=(
                "예: Existing facility datasheet / supplier property table"
            ),
            key="phase3_baseline_source",
        )

        baseline_data_ready = (
            bool(baseline_name.strip())
            and bool(baseline_source.strip())
            and baseline_rho > 0
            and baseline_cp > 0
            and baseline_mu_pa_s > 0
        )

        if not baseline_data_ready:
            st.warning(
                "Custom Baseline을 사용하려면 이름, 출처 및 "
                "Density / Cp / Viscosity를 모두 입력해야 합니다."
            )

    baseline_cols = st.columns(4)

    baseline_cols[0].metric(
        "Density",
        f"{baseline_rho:.2f} kg/m³",
    )

    baseline_cols[1].metric(
        "Cp",
        f"{baseline_cp:.3f} kJ/kg·K",
    )

    baseline_cols[2].metric(
        "Viscosity",
        f"{baseline_mu_pa_s * 1000:.3f} mPa·s",
    )

    baseline_cols[3].metric(
        "Property Temp.",
        f"{baseline_temp:.1f} °C",
    )

    st.markdown("##### Reference & Source")

    st.write(
        f"**Baseline basis:** {baseline_name} · "
        f"{baseline_temp:.1f} °C property reference"
    )

    if baseline_mode == "Default Pure-water Reference":
        st.markdown(
            """
            **Source basis:**  
            NIST · *Reference Correlations for Thermophysical Properties of Liquid Water at 0.1 MPa*  
            Based primarily on IAPWS formulations.

            - [NIST Reference Correlation](https://www.nist.gov/publications/reference-correlations-thermophysical-properties-liquid-water-01-mpa)
            - [NIST Chemistry WebBook · Fluid Properties](https://webbook.nist.gov/chemistry/fluid/)
            """
        )

        st.caption(
            "A는 순수 물의 thermophysical reference baseline이며, "
            "실제 D2C coolant의 OEM 승인 또는 적용 적합성을 의미하지 않습니다."
        )

    else:
        st.write(
            f"**User-provided source:** {baseline_source}"
        )

        st.caption(
            "Custom Baseline은 엔지니어가 입력한 프로젝트 기준 데이터입니다."
        )
    # -----------------------------------
    # B · Project Candidate
    # -----------------------------------
    st.markdown("#### B · Project Coolant Candidate")

    render_tag("PROJECT INPUT", "input")

    # ===================================
    # IMPORT ENGINEER-VERIFIED AI DATA
    # ===================================
    if "ai_phase3_import_pending" in st.session_state:
        ai_import = st.session_state.pop(
            "ai_phase3_import_pending"
        )

        st.session_state["phase3_supplier_name"] = (
            ai_import.get("name") or ""
        )

        st.session_state["phase3_supplier_rho"] = float(
            ai_import.get("rho_kg_m3") or 0.0
        )

        st.session_state["phase3_supplier_cp"] = float(
            ai_import.get("cp_kj_kgk") or 0.0
        )

        st.session_state["phase3_supplier_mu_mpas"] = float(
            ai_import.get("viscosity_mpas") or 0.0
        )

        if ai_import.get("property_temp_c") is not None:
            st.session_state[
                "phase3_supplier_property_temp"
            ] = float(
                ai_import["property_temp_c"]
            )

        st.session_state["phase3_supplier_source"] = (
            ai_import.get("source") or ""
        )

        st.session_state["phase3_ai_imported"] = True

    st.caption(
        "OEM 또는 coolant supplier 자료에서 확인한 값을 직접 입력합니다. "
        "초기값 0은 미입력 상태를 의미하며 H-LiquidOpt가 물성을 생성하지 않습니다."
    )
    if st.session_state.get(
        "phase3_ai_imported",
        False,
    ):
        st.success(
            "✓ Engineer-verified AI specification loaded. "
            "Review or edit the values below before Phase 3 approval."
        )

    candidate_name = st.text_input(
        "Coolant / Formulation Name",
        placeholder="예: Supplier Product ABC",
        key="phase3_supplier_name",
    )

    cB1, cB2, cB3 = st.columns(3)

    with cB1:
        candidate_rho = st.number_input(
            "Candidate Density (kg/m³)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="phase3_supplier_rho",
        )

    with cB2:
        candidate_cp = st.number_input(
            "Candidate Cp (kJ/kg·K)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key="phase3_supplier_cp",
        )

    with cB3:
        candidate_mu_mpas = st.number_input(
            "Candidate Viscosity (mPa·s)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key="phase3_supplier_mu_mpas",
        )

    candidate_temp = st.number_input(
        "Candidate Property Reference Temperature (°C)",
        value=float(mean_fluid_temp),
        step=1.0,
        key="phase3_supplier_property_temp",
    )

    candidate_source = st.text_input(
        "Supplier / OEM Property Source",
        placeholder=(
            "예: Supplier datasheet revision / OEM approval document"
        ),
        key="phase3_supplier_source",
    )

    candidate_data_complete = (
        bool(candidate_name.strip())
        and candidate_rho > 0
        and candidate_cp > 0
        and candidate_mu_mpas > 0
    )

    candidate_mu_pa_s = (
        candidate_mu_mpas / 1000.0
        if candidate_mu_mpas > 0
        else 0.0
    )

    if candidate_data_complete:
        st.success(
            "Candidate numerical property data · COMPLETE"
        )
    else:
        st.warning(
            "Candidate numerical property data · INCOMPLETE"
        )

    # -----------------------------------
    # C · Sensitivity Case
    # -----------------------------------
    st.markdown("#### C · PG30 Sensitivity Case")

    pg_match = cdf[
        cdf["name"] == "PG30 sensitivity fluid"
    ]

    if pg_match.empty:
        st.error(
            "coolants.csv에서 PG30 sensitivity fluid를 찾을 수 없습니다."
        )
        st.stop()

    pg_row = pg_match.iloc[0]

    sensitivity_name = "PG30 Sensitivity"

    sensitivity_rho = float(
        pg_row["rho_kg_m3"]
    )

    sensitivity_cp = float(
        pg_row["cp_kj_kgk"]
    )

    sensitivity_mu_pa_s = float(
        pg_row["mu_pa_s"]
    )

    if (
        "property_temp_c" in pg_row.index
        and pd.notna(pg_row["property_temp_c"])
    ):
        sensitivity_temp = float(
            pg_row["property_temp_c"]
        )
    else:
        sensitivity_temp = 40.0

    if (
        "source" in pg_row.index
        and pd.notna(pg_row["source"])
    ):
        sensitivity_source = str(
            pg_row["source"]
        )
    else:
        sensitivity_source = (
            "H-LiquidOpt sensitivity dataset"
        )

    render_tag("SENSITIVITY", "assumption")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Density",
        f"{sensitivity_rho:.2f} kg/m³",
    )

    s2.metric(
        "Cp",
        f"{sensitivity_cp:.3f} kJ/kg·K",
    )

    s3.metric(
        "Viscosity",
        f"{sensitivity_mu_pa_s * 1000:.3f} mPa·s",
    )

    s4.metric(
        "Property Temp.",
        f"{sensitivity_temp:.1f} °C",
    )

    st.warning(
        "C는 실제 coolant 선정안이 아니라 물성 변화에 따른 "
        "Hydraulic 영향 확인용 sensitivity case입니다."
    )

    st.caption(
        f"Sensitivity data source: {sensitivity_source}"
    )

    # ===================================
    # 3C · PROPERTY COMPARISON
    # ===================================
    st.divider()

    st.markdown("### 3C · Candidate Property Comparison")

    p1, p2 = st.columns([1, 4])

    with p1:
        render_tag("CALCULATED", "calculated")

    with p2:
        st.write(
            "동일한 Liquid Heat Load와 ΔT 조건에서 "
            "각 유체의 물성 차이와 이론 요구유량 차이를 비교합니다."
        )

    # -----------------------------------
    # Flow calculation helper
    # Q = m_dot * Cp * DeltaT
    # -----------------------------------
    def required_flow_lpm(
        heat_kw,
        rho,
        cp,
        d_t,
    ):
        if (
            heat_kw <= 0
            or rho <= 0
            or cp <= 0
            or d_t <= 0
        ):
            return 0.0

        mass_flow_kg_s = (
            heat_kw
            / (cp * d_t)
        )

        volume_flow_m3_s = (
            mass_flow_kg_s
            / rho
        )

        return volume_flow_m3_s * 60000.0

    baseline_flow_lpm = required_flow_lpm(
        total_liquid_kw,
        baseline_rho,
        baseline_cp,
        delta_t,
    )

    sensitivity_flow_lpm = required_flow_lpm(
        total_liquid_kw,
        sensitivity_rho,
        sensitivity_cp,
        delta_t,
    )

    comparison_rows = [
        {
            "Case": "A · Baseline",
            "Fluid": baseline_name,
            "Density kg/m³": baseline_rho,
            "Cp kJ/kg·K": baseline_cp,
            "Viscosity mPa·s": baseline_mu_pa_s * 1000,
            "Property Temp °C": baseline_temp,
            "Required Flow LPM": baseline_flow_lpm,
            "Flow vs Baseline %": 0.0,
            "Viscosity Ratio vs A": 1.0,
        }
    ]

    candidate_flow_lpm = 0.0

    if candidate_data_complete:
        candidate_flow_lpm = required_flow_lpm(
            total_liquid_kw,
            candidate_rho,
            candidate_cp,
            delta_t,
        )

        candidate_flow_delta = (
            (
                candidate_flow_lpm
                / baseline_flow_lpm
                - 1
            )
            * 100
            if baseline_flow_lpm > 0
            else 0.0
        )

        candidate_viscosity_ratio = (
            candidate_mu_pa_s
            / baseline_mu_pa_s
            if baseline_mu_pa_s > 0
            else 0.0
        )

        comparison_rows.append(
            {
                "Case": "B · Project Candidate",
                "Fluid": candidate_name,
                "Density kg/m³": candidate_rho,
                "Cp kJ/kg·K": candidate_cp,
                "Viscosity mPa·s": candidate_mu_mpas,
                "Property Temp °C": candidate_temp,
                "Required Flow LPM": candidate_flow_lpm,
                "Flow vs Baseline %": candidate_flow_delta,
                "Viscosity Ratio vs A": candidate_viscosity_ratio,
            }
        )

    sensitivity_flow_delta = (
        (
            sensitivity_flow_lpm
            / baseline_flow_lpm
            - 1
        )
        * 100
        if baseline_flow_lpm > 0
        else 0.0
    )

    sensitivity_viscosity_ratio = (
        sensitivity_mu_pa_s
        / baseline_mu_pa_s
        if baseline_mu_pa_s > 0
        else 0.0
    )

    comparison_rows.append(
        {
            "Case": "C · Sensitivity",
            "Fluid": sensitivity_name,
            "Density kg/m³": sensitivity_rho,
            "Cp kJ/kg·K": sensitivity_cp,
            "Viscosity mPa·s": sensitivity_mu_pa_s * 1000,
            "Property Temp °C": sensitivity_temp,
            "Required Flow LPM": sensitivity_flow_lpm,
            "Flow vs Baseline %": sensitivity_flow_delta,
            "Viscosity Ratio vs A": sensitivity_viscosity_ratio,
        }
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "Density kg/m³": "{:.2f}",
                "Cp kJ/kg·K": "{:.3f}",
                "Viscosity mPa·s": "{:.3f}",
                "Property Temp °C": "{:.1f}",
                "Required Flow LPM": "{:.1f}",
                "Flow vs Baseline %": "{:+.1f}%",
                "Viscosity Ratio vs A": "{:.2f}×",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------
    # Relative property comparison
    # -----------------------------------
    st.markdown("#### Relative to Baseline")

    def percent_change(
        value,
        baseline,
    ):
        if baseline == 0:
            return 0.0

        return (
            (value / baseline) - 1
        ) * 100

    relative_rows = []

    if candidate_data_complete:
        relative_rows.append(
            {
                "Case": "B · Project Candidate",
                "Density Δ %": percent_change(
                    candidate_rho,
                    baseline_rho,
                ),
                "Cp Δ %": percent_change(
                    candidate_cp,
                    baseline_cp,
                ),
                "Viscosity Δ %": percent_change(
                    candidate_mu_pa_s,
                    baseline_mu_pa_s,
                ),
                "Required Flow Δ %": percent_change(
                    candidate_flow_lpm,
                    baseline_flow_lpm,
                ),
            }
        )

    relative_rows.append(
        {
            "Case": "C · Sensitivity",
            "Density Δ %": percent_change(
                sensitivity_rho,
                baseline_rho,
            ),
            "Cp Δ %": percent_change(
                sensitivity_cp,
                baseline_cp,
            ),
            "Viscosity Δ %": percent_change(
                sensitivity_mu_pa_s,
                baseline_mu_pa_s,
            ),
            "Required Flow Δ %": percent_change(
                sensitivity_flow_lpm,
                baseline_flow_lpm,
            ),
        }
    )

    relative_df = pd.DataFrame(
        relative_rows
    )

    st.dataframe(
        relative_df.style.format(
            {
                "Density Δ %": "{:+.1f}%",
                "Cp Δ %": "{:+.1f}%",
                "Viscosity Δ %": "{:+.1f}%",
                "Required Flow Δ %": "{:+.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "※ Required Flow는 동일 Liquid Heat Load와 동일 ΔT 조건에서 "
        "Density와 Cp 차이에 의해 발생하는 이론 요구유량 비교입니다. "
        "배관 압력손실과 Pump Power 비교는 Phase 4에서 수행합니다."
    )

    # -----------------------------------
    # Property temperature consistency
    # -----------------------------------
    st.markdown("#### Property Temperature Check")

    temp_warning = False

    if abs(
        baseline_temp - mean_fluid_temp
    ) > 2.0:
        temp_warning = True

        st.warning(
            f"A Baseline 물성 기준온도는 {baseline_temp:.1f}°C이고 "
            f"현재 평균 유체온도는 {mean_fluid_temp:.1f}°C입니다. "
            "가능하면 실제 설계온도에 가까운 물성값을 사용하세요."
        )

    if candidate_data_complete:
        if abs(
            candidate_temp - baseline_temp
        ) > 1.0:
            temp_warning = True

            st.warning(
                f"A Baseline은 {baseline_temp:.1f}°C, "
                f"B Project Candidate는 {candidate_temp:.1f}°C 물성입니다. "
                "동일 온도 기준의 supplier property 확보를 권장합니다."
            )

    if abs(
        sensitivity_temp - baseline_temp
    ) > 1.0:
        temp_warning = True

        st.warning(
            f"A Baseline은 {baseline_temp:.1f}°C, "
            f"C Sensitivity는 {sensitivity_temp:.1f}°C 물성입니다. "
            "결과 해석 시 기준온도 차이를 고려해야 합니다."
        )

    if not temp_warning:
        st.success(
            "Property reference temperatures are aligned."
        )

    # ===================================
    # 3D · MATERIAL & EVIDENCE VALIDATION
    # ===================================
    st.divider()

    st.markdown(
        "### 3D · Material & Evidence Validation"
    )

    d1, d2 = st.columns([1, 4])

    with d1:
        render_tag(
            "REVIEW REQUIRED",
            "review",
        )

    with d2:
        st.write(
            "Project Candidate의 실제 적용 가능성을 자동 확정하지 않고, "
            "wetted material 구성과 OEM / supplier 근거 확보 여부를 확인합니다."
        )

    st.markdown("#### Wetted Materials")

    wetted_materials = st.multiselect(
        "Materials in contact with coolant",
        [
            "Copper / Copper Alloy",
            "Stainless Steel",
            "Aluminum / Aluminum Alloy",
            "EPDM",
            "FKM",
            "NBR",
            "Engineering Plastics",
            "Other",
        ],
        default=[
            "Copper / Copper Alloy",
            "Stainless Steel",
            "EPDM",
        ],
        key="phase3_wetted_materials",
    )

    if (
        "Copper / Copper Alloy" in wetted_materials
        and "Aluminum / Aluminum Alloy" in wetted_materials
    ):
        st.warning(
            "Cu계와 Al계 재질이 동일 coolant loop에 포함되어 있습니다. "
            "자동 Reject하지 않으며 galvanic corrosion 위험, inhibitor 구성 및 "
            "supplier compatibility 자료를 별도로 검토해야 합니다."
        )

    st.caption(
        "※ 재질 목록만으로 compatibility를 확정하지 않습니다. "
        "최종 판단에는 coolant supplier의 compatibility 자료와 "
        "실제 재질·온도·농도 조건 검토가 필요합니다."
    )

    st.markdown(
        "#### Project Candidate Evidence"
    )

    if candidate_data_complete:

        check_oem = st.checkbox(
            "OEM이 해당 coolant / formulation의 사용을 허용함을 확인했습니다.",
            key="phase3_check_oem",
        )

        check_properties = st.checkbox(
            "입력한 Density, Cp, Viscosity가 supplier/OEM 자료와 일치함을 확인했습니다.",
            key="phase3_check_properties",
        )

        check_material = st.checkbox(
            "선택한 wetted materials에 대한 compatibility 자료를 확인했습니다.",
            key="phase3_check_material",
        )

        source_ready = bool(
            candidate_source.strip()
        )

        candidate_evidence_ready = (
            check_oem
            and check_properties
            and check_material
            and source_ready
        )

        if not source_ready:
            st.warning(
                "Supplier / OEM Property Source를 입력해야 합니다."
            )

        if candidate_evidence_ready:
            st.success(
                "B Project Candidate · ELIGIBLE FOR PHASE 4 ANALYSIS"
            )
        else:
            st.warning(
                "B Project Candidate · EVIDENCE INCOMPLETE"
            )

    else:
        candidate_evidence_ready = False

        st.info(
            "B Project Candidate의 물성값이 완성되지 않아 "
            "현재 Phase 4 계산 대상으로 사용할 수 없습니다."
        )

    # ===================================
    # 3E · ENGINEER REVIEW
    # ===================================
    st.divider()

    st.markdown("### 3E · Engineer Review")

    e1, e2 = st.columns([1, 4])

    with e1:
        render_tag(
            "REVIEW REQUIRED",
            "review",
        )

    with e2:
        st.write(
            "Baseline, Project Candidate 및 Sensitivity Case의 역할과 "
            "데이터 근거를 확인한 후 Phase 4 계산 대상을 승인합니다."
        )

    analysis_options = [
        "A · Baseline Reference",
        "C · PG30 Sensitivity",
    ]

    if (
        candidate_data_complete
        and candidate_evidence_ready
    ):
        analysis_options.insert(
            1,
            "B · Project Candidate",
        )

    selected_analysis_cases = st.multiselect(
        "Cases to send to Phase 4",
        analysis_options,
        default=analysis_options,
        key="phase3_analysis_cases",
    )

    check_comparison = st.checkbox(
        "A/B/C 물성 및 요구유량 비교 결과를 확인했습니다.",
        key="phase3_check_comparison",
    )

    check_role = st.checkbox(
        "A는 Baseline, B는 실제 Project Candidate, C는 Sensitivity Case임을 확인했습니다.",
        key="phase3_check_role",
    )

    check_limit = st.checkbox(
        "Phase 3 결과는 coolant의 최종 적합성 또는 최적 농도를 자동 결정하지 않음을 확인했습니다.",
        key="phase3_check_limit",
    )

    phase3_note = st.text_area(
        "Engineer note",
        key="phase3_note",
        placeholder=(
            "예: Project Candidate B의 요구유량이 Baseline 대비 증가함. "
            "Phase 4에서 Pressure Drop 및 Pump Power 영향을 추가 비교."
        ),
    )

    ready_phase3 = (
        baseline_data_ready
        and len(selected_analysis_cases) > 0
        and check_comparison
        and check_role
        and check_limit
    )

    if st.button(
        "✓ Approve Phase 3 Analysis Cases",
        type="primary",
        disabled=not ready_phase3,
        use_container_width=True,
    ):

        # -----------------------------------
        # Save Baseline
        # -----------------------------------
        st.session_state.phase3_baseline = {
            "name": baseline_name,
            "rho_kg_m3": float(
                baseline_rho
            ),
            "cp_kj_kgk": float(
                baseline_cp
            ),
            "mu_pa_s": float(
                baseline_mu_pa_s
            ),
            "property_temp_c": float(
                baseline_temp
            ),
            "source": baseline_source,
        }

        # -----------------------------------
        # Save Sensitivity Case
        # -----------------------------------
        st.session_state.phase3_sensitivity = {
            "name": sensitivity_name,
            "rho_kg_m3": float(
                sensitivity_rho
            ),
            "cp_kj_kgk": float(
                sensitivity_cp
            ),
            "mu_pa_s": float(
                sensitivity_mu_pa_s
            ),
            "property_temp_c": float(
                sensitivity_temp
            ),
            "source": sensitivity_source,
        }

        # -----------------------------------
        # Save Project Candidate
        # -----------------------------------
        if (
            candidate_data_complete
            and candidate_evidence_ready
        ):
            st.session_state.phase3_supplier_coolant = {
                "name": candidate_name.strip(),
                "rho_kg_m3": float(
                    candidate_rho
                ),
                "cp_kj_kgk": float(
                    candidate_cp
                ),
                "mu_pa_s": float(
                    candidate_mu_pa_s
                ),
                "property_temp_c": float(
                    candidate_temp
                ),
                "source": candidate_source.strip(),
            }

        st.session_state.phase3_approved_analysis_cases = (
            list(selected_analysis_cases)
        )

        st.session_state.phase3_supply_t = (
            supply_t
        )

        st.session_state.phase3_return_t = (
            return_t
        )

        st.session_state.phase3_approved_wetted_materials = (
            list(wetted_materials)
        )

        # -----------------------------------
        # Temporary compatibility with
        # existing Phase 4
        # -----------------------------------
        coolant_names = []

        if (
            "A · Baseline Reference"
            in selected_analysis_cases
            and baseline_mode
            == "Default Pure-water Reference"
        ):
            coolant_names.append(
                "Water-based reference"
            )

        if (
            "C · PG30 Sensitivity"
            in selected_analysis_cases
        ):
            coolant_names.append(
                "PG30 sensitivity fluid"
            )

        st.session_state.coolant_names = (
            coolant_names
        )

        st.session_state.approved[3] = True

        st.success(
            "Phase 3 approved. "
            "선택된 coolant analysis cases가 Phase 4로 전달되었습니다."
        )

    if st.session_state.approved[3]:
        st.success(
            "✓ Phase 3 Engineer Review Approved"
        )

elif phase == 4:
    st.header("Phase 4 · Deterministic Hydraulic Physics Engine")

    st.caption(
        "Phase 1~3에서 승인된 Heat Load, Coolant 물성 및 설계조건을 이용해 "
        "Required Flow → Velocity → Reynolds / Friction → Pressure Drop → "
        "Pump Power를 결정론적으로 계산합니다."
    )

    # ===================================
    # PHASE 3 APPROVAL GATE
    # ===================================
    if not st.session_state.approved[3]:
        st.warning(
            "Phase 3 Engineer Review가 아직 완료되지 않았습니다. "
            "Coolant analysis case를 승인한 후 Hydraulic 계산을 진행해주세요."
        )
        st.stop()

    if "phase1_racks" not in st.session_state:
        st.error(
            "승인된 Phase 1 Rack Load Model이 없습니다."
        )
        st.stop()

    if "phase3_approved_analysis_cases" not in st.session_state:
        st.error(
            "Phase 3에서 승인된 coolant analysis case가 없습니다."
        )
        st.stop()

    phase4_racks = (
        st.session_state.phase1_racks.copy()
    )

    selected_cases = (
        st.session_state.phase3_approved_analysis_cases
    )

    phase4_supply_t = (
        st.session_state.get(
            "phase3_supply_t",
            supply_t,
        )
    )

    phase4_return_t = (
        st.session_state.get(
            "phase3_return_t",
            return_t,
        )
    )

    phase4_delta_t = (
        phase4_return_t
        - phase4_supply_t
    )

    if phase4_delta_t <= 0:
        st.error(
            "승인된 Return Temperature는 Supply Temperature보다 높아야 합니다."
        )
        st.stop()

    # ===================================
    # BUILD APPROVED COOLANT INPUTS
    # ===================================
    coolants = []

    # A · Baseline
    if (
        "A · Baseline Reference"
        in selected_cases
    ):
        baseline_data = (
            st.session_state.get(
                "phase3_baseline"
            )
        )

        if baseline_data:
            coolants.append(
                Coolant(
                    baseline_data["name"],
                    float(
                        baseline_data["rho_kg_m3"]
                    ),
                    float(
                        baseline_data["cp_kj_kgk"]
                    ),
                    float(
                        baseline_data["mu_pa_s"]
                    ),
                    "Baseline Reference",
                )
            )

    # B · Project Candidate
    if (
        "B · Project Candidate"
        in selected_cases
    ):
        candidate_data = (
            st.session_state.get(
                "phase3_supplier_coolant"
            )
        )

        if candidate_data:
            coolants.append(
                Coolant(
                    candidate_data["name"],
                    float(
                        candidate_data["rho_kg_m3"]
                    ),
                    float(
                        candidate_data["cp_kj_kgk"]
                    ),
                    float(
                        candidate_data["mu_pa_s"]
                    ),
                    "Project Candidate",
                )
            )
        else:
            st.warning(
                "B Project Candidate가 선택되었지만 "
                "승인된 numerical property data를 찾을 수 없습니다."
            )

    # C · Sensitivity
    if (
        "C · PG30 Sensitivity"
        in selected_cases
    ):
        sensitivity_data = (
            st.session_state.get(
                "phase3_sensitivity"
            )
        )

        if sensitivity_data:
            coolants.append(
                Coolant(
                    sensitivity_data["name"],
                    float(
                        sensitivity_data["rho_kg_m3"]
                    ),
                    float(
                        sensitivity_data["cp_kj_kgk"]
                    ),
                    float(
                        sensitivity_data["mu_pa_s"]
                    ),
                    "Sensitivity Case",
                )
            )

    if not coolants:
        st.error(
            "Phase 4에서 계산 가능한 승인 coolant case가 없습니다."
        )
        st.stop()

    # ===================================
    # 4A · HYDRAULIC DESIGN BASIS
    # ===================================
    st.markdown(
        "### 4A · Hydraulic Design Basis"
    )

    b1, b2 = st.columns([1, 4])

    with b1:
        render_tag(
            "PHASE 3 APPROVED",
            "verified",
        )

    with b2:
        st.write(
            "Thermal Required Flow는 사용자가 임의 입력하지 않고 "
            "Liquid Heat Load, Coolant Cp·Density 및 ΔT로 계산합니다."
        )

    phase4_heat = heat_loads(
        phase4_racks
    )

    total_liquid_kw = (
        phase4_heat[
            "liquid_load_kw"
        ].sum()
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Liquid Heat Load",
        f"{total_liquid_kw / 1000:.2f} MW",
    )

    m2.metric(
        "Design ΔT",
        f"{phase4_delta_t:.1f} K",
    )

    m3.metric(
        "Coolant Cases",
        f"{len(coolants)}",
    )

    m4.metric(
        "Approved Topology",
        st.session_state.get(
            "topology_choice",
            "Not selected",
        ),
    )

    st.info(
        "계산 흐름: Heat Load → Required Flow → Pipe Velocity → "
        "Reynolds / Friction Factor → Pressure Drop → Pump Power"
    )

    # ===================================
    # 4B · PIPE / COMPONENT INPUT REVIEW
    # ===================================
    st.divider()

    st.markdown(
        "### 4B · Pipe & Component Inputs"
    )

    p1, p2 = st.columns([1, 4])

    with p1:
        render_tag(
            "USER INPUT",
            "input",
        )

    with p2:
        st.write(
            "Phase 4 Sidebar에서 입력한 배관 직경, 길이 및 Rack ΔP 가정을 "
            "Hydraulic 계산의 geometry 조건으로 사용합니다."
        )

    geometry_table = pd.DataFrame(
        [
            {
                "Section": "Common Pipe",
                "Diameter m": st.session_state.get(
                    "common_d",
                    0.2027,
                ),
                "Length m": st.session_state.get(
                    "common_l",
                    20.0,
                ),
            },
            {
                "Section": "Row Header",
                "Diameter m": st.session_state.get(
                    "row_d",
                    0.1541,
                ),
                "Length m": st.session_state.get(
                    "row_l",
                    12.0,
                ),
            },
            {
                "Section": "Rack Branch",
                "Diameter m": st.session_state.get(
                    "branch_d",
                    0.0525,
                ),
                "Length m": st.session_state.get(
                    "branch_l",
                    6.0,
                ),
            },
        ]
    )

    st.dataframe(
        geometry_table,
        use_container_width=True,
        hide_index=True,
    )

    rack_dp_assumption = float(
        st.session_state.get(
            "rack_dp",
            120.0,
        )
    )

    st.warning(
        f"현재 Rack internal ΔP = {rack_dp_assumption:.1f} kPa는 "
        "PoC용 설계 가정입니다. 실제 설계에서는 OEM Rack / Cold Plate의 "
        "pressure-flow curve로 교체해야 합니다."
    )

    # ===================================
    # 4C · DETERMINISTIC CALCULATION
    # ===================================
    st.divider()

    st.markdown(
        "### 4C · Deterministic Hydraulic Calculation"
    )

    c1, c2 = st.columns([1, 4])

    with c1:
        render_tag(
            "CALCULATED",
            "calculated",
        )

    with c2:
        st.write(
            "동일한 Rack Load와 동일 배관 Geometry에 대해 "
            "Phase 3에서 승인된 각 coolant case를 계산합니다."
        )

    results = evaluate_coolants(
        phase4_racks,
        coolants,
        phase4_delta_t,
        geom,
    )

    if results.empty:
        st.error(
            "Hydraulic calculation result가 생성되지 않았습니다."
        )
        st.stop()

    desired_cols = [
        "coolant",
        "pod",
        "liquid_racks",
        "liquid_load_kw",
        "rack_avg_heat_kw",
        "rack_flow_lpm",
        "pod_flow_lpm",
        "branch_velocity_m_s",
        "network_dp_kpa",
        "rack_dp_kpa",
        "total_dp_kpa",
        "pump_kw",
    ]

    show_cols = [
        col
        for col in desired_cols
        if col in results.columns
    ]

    st.dataframe(
        results[show_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "※ Flow는 Q = ṁCpΔT로부터 계산된 Thermal Required Flow입니다. "
        "현재 Rack ΔP는 synthetic placeholder이며, 최종 설계값이 아닙니다."
    )

    # ===================================
    # 4D · COOLANT HYDRAULIC COMPARISON
    # ===================================
    st.divider()

    st.markdown(
        "### 4D · Hydraulic Case Comparison"
    )

    hydraulic_summary = (
        results
        .groupby(
            "coolant",
            as_index=False,
        )
        .agg(
            Total_Flow_LPM=(
                "pod_flow_lpm",
                "sum",
            ),
            Max_Branch_Velocity_m_s=(
                "branch_velocity_m_s",
                "max",
            ),
            Worst_Total_DP_kPa=(
                "total_dp_kpa",
                "max",
            ),
            Total_Pump_kW=(
                "pump_kw",
                "sum",
            ),
        )
    )

    st.dataframe(
        hydraulic_summary,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------
    # Relative comparison against A
    # -----------------------------------
    baseline_data = (
        st.session_state.get(
            "phase3_baseline"
        )
    )

    if baseline_data:
        baseline_name = (
            baseline_data["name"]
        )

        base_match = hydraulic_summary[
            hydraulic_summary["coolant"]
            == baseline_name
        ]

        if not base_match.empty:
            base_row = base_match.iloc[0]

            relative_hydraulic = (
                hydraulic_summary.copy()
            )

            def relative_change(
                value,
                baseline_value,
            ):
                if baseline_value == 0:
                    return 0.0

                return (
                    value
                    / baseline_value
                    - 1
                ) * 100

            relative_hydraulic[
                "Flow Δ vs A %"
            ] = relative_hydraulic[
                "Total_Flow_LPM"
            ].apply(
                lambda x: relative_change(
                    x,
                    base_row[
                        "Total_Flow_LPM"
                    ],
                )
            )

            relative_hydraulic[
                "Worst ΔP Δ vs A %"
            ] = relative_hydraulic[
                "Worst_Total_DP_kPa"
            ].apply(
                lambda x: relative_change(
                    x,
                    base_row[
                        "Worst_Total_DP_kPa"
                    ],
                )
            )

            relative_hydraulic[
                "Pump Power Δ vs A %"
            ] = relative_hydraulic[
                "Total_Pump_kW"
            ].apply(
                lambda x: relative_change(
                    x,
                    base_row[
                        "Total_Pump_kW"
                    ],
                )
            )

            st.markdown(
                "#### Relative to Baseline A"
            )

            st.dataframe(
                relative_hydraulic[
                    [
                        "coolant",
                        "Flow Δ vs A %",
                        "Worst ΔP Δ vs A %",
                        "Pump Power Δ vs A %",
                    ]
                ].style.format(
                    {
                        "Flow Δ vs A %": "{:+.1f}%",
                        "Worst ΔP Δ vs A %": "{:+.1f}%",
                        "Pump Power Δ vs A %": "{:+.1f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------
    # Charts
    # -----------------------------------
    f1, f2 = st.columns(2)

    with f1:
        fig = px.bar(
            results,
            x="pod",
            y="pump_kw",
            color="coolant",
            barmode="group",
            title="Pump Electric Power by Pod",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with f2:
        fig2 = px.bar(
            results,
            x="pod",
            y="total_dp_kpa",
            color="coolant",
            barmode="group",
            title="Calculated Pressure Drop by Pod",
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )

    st.caption(
        "점도 증가가 Pump Power에 동일 비율로 직접 반영되는 것은 아닙니다. "
        "Coolant 물성은 Required Flow, Reynolds number, friction factor 및 "
        "Pressure Drop을 통해 복합적으로 Hydraulic 결과에 영향을 줍니다."
    )

    # ===================================
    # 4E · ENGINEERING CONSTRAINT REVIEW
    # ===================================
    st.divider()

    st.markdown(
        "### 4E · Engineering Constraint Review"
    )

    r1, r2 = st.columns([1, 4])

    with r1:
        render_tag(
            "REVIEW REQUIRED",
            "review",
        )

    with r2:
        st.write(
            "Physics Engine의 계산결과와 현재 적용된 가정의 한계를 "
            "엔지니어가 확인합니다."
        )

    st.info(
        "현재 PoC는 임의의 universal velocity / pressure-drop 기준을 "
        "자동 적용하지 않습니다. 실제 허용범위는 프로젝트 기준, "
        "OEM 요구조건 및 배관 설계기준을 통해 검증해야 합니다."
    )

    check_flow_basis = st.checkbox(
        "Thermal Required Flow가 Heat Load, Coolant 물성 및 ΔT로 계산됨을 확인했습니다.",
        key="phase4_check_flow_basis",
    )

    check_geometry = st.checkbox(
        "현재 Pipe Diameter / Length 입력조건을 확인했습니다.",
        key="phase4_check_geometry",
    )

    check_rack_dp = st.checkbox(
        "Rack internal ΔP가 현재 PoC placeholder이며 OEM pressure-flow curve로 교체해야 함을 확인했습니다.",
        key="phase4_check_rack_dp",
    )

    check_limits = st.checkbox(
        "Velocity / ΔP / Pump 운전범위의 최종 허용성은 프로젝트별 기준으로 추가 검토해야 함을 확인했습니다.",
        key="phase4_check_limits",
    )

    phase4_note = st.text_area(
        "Engineer calculation review note",
        key="phase4_note",
        placeholder=(
            "예: Project Candidate B는 Baseline 대비 Pump Power 증가. "
            "OEM Rack pressure-flow curve 확보 후 재검증 필요."
        ),
    )

    ready_phase4 = (
        check_flow_basis
        and check_geometry
        and check_rack_dp
        and check_limits
    )

    st.download_button(
        "Download hydraulic CSV",
        results.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        "hliquidopt_hydraulics.csv",
        "text/csv",
        use_container_width=True,
    )

    if st.button(
        "✓ Approve Phase 4 Calculation Review",
        type="primary",
        disabled=not ready_phase4,
        use_container_width=True,
    ):
        st.session_state.hydraulic_results = (
            results.copy()
        )

        st.session_state.hydraulic_summary = (
            hydraulic_summary.copy()
        )

        st.session_state.phase4_delta_t = (
            phase4_delta_t
        )

        st.session_state.phase4_geometry = {
            "common_d": st.session_state.get(
                "common_d"
            ),
            "row_d": st.session_state.get(
                "row_d"
            ),
            "branch_d": st.session_state.get(
                "branch_d"
            ),
            "common_l": st.session_state.get(
                "common_l"
            ),
            "row_l": st.session_state.get(
                "row_l"
            ),
            "branch_l": st.session_state.get(
                "branch_l"
            ),
            "rack_dp_kpa": rack_dp_assumption,
        }

        st.session_state.approved[4] = True

        st.success(
            "Phase 4 approved. "
            "검토된 Hydraulic 결과가 Phase 5로 전달되었습니다."
        )

    if st.session_state.approved[4]:
        st.success(
            "✓ Phase 4 Engineer Review Approved"
        )


elif phase == 5:
    st.header(
        "Phase 5 · Integrated Review & Engineer Decision"
    )

    st.caption(
        "승인된 TCS / CDU topology와 Phase 4 Hydraulic 결과를 취합하여 "
        "후속 설계검토에 사용할 preferred engineering case를 결정합니다."
    )

    # ===================================
    # PHASE 4 APPROVAL GATE
    # ===================================
    if not st.session_state.approved[4]:
        st.warning(
            "Phase 4 Engineer Review가 아직 완료되지 않았습니다. "
            "Hydraulic 계산을 승인한 후 최종 비교를 진행해주세요."
        )
        st.stop()

    if "hydraulic_results" not in st.session_state:
        st.error(
            "승인된 Hydraulic 결과가 없습니다."
        )
        st.stop()

    phase5_results = (
        st.session_state.hydraulic_results.copy()
    )

    phase5_racks = (
        st.session_state.phase1_racks.copy()
    )

    phase5_cdu_capacity = (
        st.session_state.get(
            "phase2_cdu_capacity",
            cdu_capacity,
        )
    )

    phase5_redundancy = (
        st.session_state.get(
            "phase2_redundancy",
            redundancy,
        )
    )

    phase5_delta_t = (
        st.session_state.get(
            "phase4_delta_t",
            return_t - supply_t,
        )
    )

    # ===================================
    # 5A · APPROVED DESIGN BASIS
    # ===================================
    st.markdown(
        "### 5A · Approved Design Basis"
    )

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Topology",
        st.session_state.get(
            "topology_choice",
            "Not selected",
        ),
    )

    a2.metric(
        "CDU Capacity",
        f"{phase5_cdu_capacity:.1f} MW",
    )

    a3.metric(
        "Redundancy",
        phase5_redundancy,
    )

    a4.metric(
        "ΔT",
        f"{phase5_delta_t:.1f} K",
    )

    # ===================================
    # 5B · HYDRAULIC CASE COMPARISON
    # ===================================
    st.divider()

    st.markdown(
        "### 5B · Hydraulic Case Comparison"
    )

    ranking = candidate_score_table(
        phase5_results,
        phase5_cdu_capacity,
    )

    ranking_cols = [
        "rank",
        "coolant",
        "total_pump_kw",
        "worst_dp_kpa",
        "cdu_loading_pct",
        "balanced_score",
    ]

    ranking_cols = [
        col
        for col in ranking_cols
        if col in ranking.columns
    ]

    st.dataframe(
        ranking[ranking_cols],
        use_container_width=True,
        hide_index=True,
    )

    if not ranking.empty:
        first_case = ranking.iloc[0]

        st.info(
            f"현재 정의된 hydraulic composite score에서 가장 낮은 값은 "
            f"**{first_case['coolant']}**입니다. "
            "이는 최종 coolant 추천이 아니라 현재 계산항목과 weighting에 따른 "
            "비교 결과입니다."
        )

    st.warning(
        "Phase 5의 ranking은 OEM approval, 실제 CAPEX, 동결보호, "
        "water chemistry, 장기 부식/재질 compatibility 등을 모두 포함한 "
        "최종 최적화 결과가 아닙니다."
    )

    # ===================================
    # 5C · ENGINEER DECISION
    # ===================================
    st.divider()

    st.markdown(
        "### 5C · Engineer Decision"
    )

    available_cases = (
        phase5_results[
            "coolant"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    preferred_case = st.selectbox(
        "Preferred hydraulic case for next design iteration",
        available_cases,
        key="phase5_preferred_case",
    )

    st.write(
        f"**Approved TCS topology:** "
        f"{st.session_state.get('topology_choice', 'Not selected')}"
    )

    st.write(
        f"**Selected hydraulic case:** "
        f"{preferred_case}"
    )

    final_note = st.text_area(
        "Final engineer review note",
        key="phase5_final_note",
        placeholder=(
            "예: Candidate B를 후속 상세검토 대상으로 선정. "
            "OEM pressure-flow curve, supplier compatibility 및 "
            "actual routing을 반영해 다음 iteration 수행."
        ),
    )

    final_check = st.checkbox(
        "본 결과가 기본설계 단계의 비교·검토 결과이며 최종 시공/구매 승인안이 아님을 확인했습니다.",
        key="phase5_final_check",
    )

    if st.button(
        "Save Engineer Decision",
        type="primary",
        disabled=not final_check,
        use_container_width=True,
    ):
        st.session_state.final_decision = {
            "topology": st.session_state.get(
                "topology_choice"
            ),
            "hydraulic_case": preferred_case,
            "engineer_note": final_note,
        }

        st.success(
            "Engineer decision saved."
        )

    # ===================================
    # 5D · DECISION HISTORY
    # ===================================
    st.divider()

    st.markdown(
        "### 5D · Decision History"
    )

    phase3_cases_text = ", ".join(
        st.session_state.get(
            "phase3_approved_analysis_cases",
            [],
        )
    )

    hist = pd.DataFrame(
        [
            [
                "Phase 1",
                "Rack / Heat-load Model",
                "Approved"
                if st.session_state.approved[1]
                else "Pending",
                st.session_state.get(
                    "phase1_note",
                    "",
                ),
            ],
            [
                "Phase 2",
                st.session_state.get(
                    "topology_choice",
                    "Not selected",
                ),
                "Approved"
                if st.session_state.approved[2]
                else "Pending",
                st.session_state.get(
                    "phase2_note",
                    "",
                ),
            ],
            [
                "Phase 3",
                phase3_cases_text,
                "Approved"
                if st.session_state.approved[3]
                else "Pending",
                st.session_state.get(
                    "phase3_note",
                    "",
                ),
            ],
            [
                "Phase 4",
                "Deterministic Hydraulic Calculation",
                "Approved"
                if st.session_state.approved[4]
                else "Pending",
                st.session_state.get(
                    "phase4_note",
                    "",
                ),
            ],
        ],
        columns=[
            "Phase",
            "Decision / Result",
            "Status",
            "Engineer Note",
        ],
    )

    st.dataframe(
        hist,
        use_container_width=True,
        hide_index=True,
    )

    # ===================================
    # 5E · REPORT EXPORT
    # ===================================
    st.divider()

    st.markdown(
        "### 5E · Design Review Export"
    )

    pods = pod_summary(
        phase5_racks
    )

    phase5_names = (
        phase5_results[
            "coolant"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    report = project_report_markdown(
        phase5_racks,
        pods,
        ranking,
        phase5_names,
        phase5_delta_t,
        phase5_cdu_capacity,
        phase5_redundancy,
    )

    d1, d2 = st.columns(2)

    d1.download_button(
        "Download design-review report (.md)",
        report.encode(
            "utf-8-sig"
        ),
        "H-LiquidOpt_design_review.md",
        "text/markdown",
        use_container_width=True,
    )

    d2.download_button(
        "Download approved rack dataset (.csv)",
        phase5_racks.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        "H-LiquidOpt_approved_racks.csv",
        "text/csv",
        use_container_width=True,
    )

st.divider()
st.caption("Prototype only · Not for construction, procurement, safety certification, or final equipment/coolant selection. Project-specific constraints must be verified by qualified engineers and equipment/coolant suppliers.")
