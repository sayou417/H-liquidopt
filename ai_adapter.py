import base64
import io
import json
import os

import openai
from openai import OpenAI
from pypdf import PdfReader


DEFAULT_MODEL = "gpt-5.6-luna"
MAX_PDF_BYTES = 50 * 1024 * 1024

# =========================================================
# OpenAI Client
# =========================================================
def get_openai_client(api_key=None):
    """
    Create an OpenAI client.

    API key must come from Streamlit Secrets
    or environment variables.
    Never hard-code API keys.
    """

    key = api_key or os.getenv("OPENAI_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(api_key=key)


# =========================================================
# Connection Test
# =========================================================
def test_openai_connection(api_key=None):
    """
    Minimal API connection test.
    No engineering calculation is performed here.
    """

    client = get_openai_client(api_key)

    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=(
            "Reply with exactly this text and nothing else: "
            "H-LiquidOpt AI connected"
        ),
    )

    return response.output_text.strip()


# =========================================================
# Structured Output Schema
# =========================================================
SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": ["string", "null"]
        },

        "manufacturer": {
            "type": ["string", "null"]
        },

        "model": {
            "type": ["string", "null"]
        },

        "rated_power_kw": {
            "type": ["number", "null"]
        },

        "hcr": {
            "type": ["number", "null"]
        },

        "coolant_name": {
            "type": ["string", "null"]
        },

        "density_kg_m3": {
            "type": ["number", "null"]
        },

        "cp_kj_kgk": {
            "type": ["number", "null"]
        },

        "viscosity_mpas": {
            "type": ["number", "null"]
        },

        "property_temp_c": {
            "type": ["number", "null"]
        },

        "supply_temp_min_c": {
            "type": ["number", "null"]
        },

        "supply_temp_max_c": {
            "type": ["number", "null"]
        },

        "recommended_flow_lpm": {
            "type": ["number", "null"]
        },

        "pressure_drop_kpa": {
            "type": ["number", "null"]
        },
        
        "rack_flow_pressure_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "flow_lpm": {
                        "type": "number"
                    },
                    "pressure_drop_kpa": {
                        "type": "number"
                    },
                    "page": {
                        "type": [
                            "integer",
                            "null",
                        ]
                    },
                    "evidence": {
                        "type": "string"
                    },
                    "condition": {
                        "type": [
                            "string",
                            "null",
                        ]
                    },
                },
                "required": [
                    "flow_lpm",
                    "pressure_drop_kpa",
                    "page",
                    "evidence",
                    "condition",
                ],
                "additionalProperties": False,
            },
        },
        
        "notes": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string"
                    },

                    "page": {
                        "type": ["integer", "null"]
                    },

                    "evidence": {
                        "type": ["string", "null"]
                    }
                },

                "required": [
                    "field",
                    "page",
                    "evidence"
                ],

                "additionalProperties": False
            }
        }
    },

    "required": [
        "document_type",
        "manufacturer",
        "model",
        "rated_power_kw",
        "hcr",
        "coolant_name",
        "density_kg_m3",
        "cp_kj_kgk",
        "viscosity_mpas",
        "property_temp_c",
        "supply_temp_min_c",
        "supply_temp_max_c",
        "recommended_flow_lpm",
        "pressure_drop_kpa",
        "rack_flow_pressure_points",
        "notes",
        "sources"
    ],

    "additionalProperties": False
}

# =========================================================
# PDF Validation
# =========================================================
def validate_pdf(file_bytes, filename):
    """
    Validate an uploaded PDF before sending it to the AI API.
    """

    if not filename:
        raise ValueError(
            "File name is missing."
        )

    if not filename.lower().endswith(".pdf"):
        raise ValueError(
            "Only PDF files are supported."
        )

    if not file_bytes:
        raise ValueError(
            "The uploaded PDF is empty."
        )

    if len(file_bytes) >= MAX_PDF_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)

        raise ValueError(
            f"The PDF is {size_mb:.1f} MB. "
            "Please upload a file smaller than 50 MB."
        )

    if not file_bytes.startswith(b"%PDF"):
        raise ValueError(
            "The uploaded file does not appear to be a valid PDF."
        )

    try:
        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

    except Exception as exc:
        raise ValueError(
            "The PDF appears to be damaged or unreadable."
        ) from exc

    if reader.is_encrypted:
        raise ValueError(
            "Password-protected or encrypted PDFs are not supported. "
            "Please upload an unlocked copy."
        )

    if len(reader.pages) == 0:
        raise ValueError(
            "The PDF contains no readable pages."
        )

    return {
        "page_count": len(reader.pages),
        "size_mb": len(file_bytes) / (1024 * 1024),
    }
# =========================================================
# AI Specification Extraction
# =========================================================
def extract_specification(
    file_bytes,
    filename,
    api_key=None,
):
    """
    Extract engineering specification candidates
    from an uploaded PDF.

    IMPORTANT:
    This function extracts candidate values only.

    It does NOT:
    - approve engineering values
    - calculate missing engineering values
    - recommend a coolant
    - determine final equipment suitability
    """

    validate_pdf(
        file_bytes,
        filename,
    )

    client = get_openai_client(api_key)

    encoded_file = (
        base64.b64encode(file_bytes)
        .decode("utf-8")
    )

    instructions = """
You are the AI Specification Assistant for H-LiquidOpt,
a preliminary D2C liquid-cooling engineering design-support system.

Your ONLY job is to extract engineering specification candidates
that are explicitly supported by the uploaded source document.

IMPORTANT RULES

1. Never invent or estimate a value.

2. If a value is missing, unclear, conditional,
   or cannot be confidently associated with the equipment,
   return null.

3. Do NOT calculate missing values from other values.

4. Do NOT infer HCR.

5. Do NOT infer coolant density, specific heat,
   viscosity, flow, pressure drop, or operating temperature.

6. Do NOT decide whether a coolant, CDU, rack,
   or server is suitable for a project.

7. Do NOT recommend an optimum coolant.

8. Do NOT perform hydraulic calculations.

9. HCR must be returned as a decimal from 0 to 1.
   Example:
   85 percent -> 0.85

10. Density must be returned in kg/m3.

11. Specific heat must be returned in kJ/(kg K).

12. Dynamic viscosity must be returned in mPa·s.

13. Flow must be returned in L/min.

14. Pressure drop must be returned in kPa.

15. Temperatures must be returned in degrees Celsius.

16. For each meaningful extracted field,
    add a source record identifying the field
    and source page when identifiable.

17. Evidence must be a short identifying phrase only.
    Do not reproduce long passages from the source.

18. If the document contains multiple operating conditions,
    do not combine unrelated conditions.

19. If a property is stated at a particular temperature,
    preserve that property reference temperature.

20. Any ambiguity or limitation that an engineer should review
    should be placed in the notes field.

RACK FLOW-PRESSURE CURVE EXTRACTION

If the source document contains multiple paired operating points
for rack liquid flow and rack internal pressure drop, extract them
into rack_flow_pressure_points.

Examples of acceptable source structures:
- a table containing Flow Rate and Pressure Drop
- multiple explicitly stated operating points
- a rack pressure-flow performance table

Rules:

1. Extract only explicitly documented paired values.

2. Do NOT generate additional points by interpolation,
   extrapolation, regression, or engineering estimation.

3. Do NOT derive a pressure-drop curve yourself.
   Curve fitting is performed later by the deterministic
   H-LiquidOpt physics engine.

4. The pressure drop must refer to the rack internal liquid path,
   cold-plate/rack liquid circuit, or an equivalent clearly
   documented rack hydraulic boundary.

5. Do NOT mix system pressure drop, CDU pressure drop,
   pump head, facility piping pressure drop, or unrelated
   hydraulic boundaries with rack internal pressure drop.

6. When possible, record the page and a short evidence statement
   for each operating point.

7. If the operating points are associated with a coolant,
   temperature, or other test condition, preserve that information
   in the condition field.

8. Do NOT combine points from different equipment models,
   different coolant formulations, or clearly different test
   conditions into one curve unless the source explicitly presents
   them as one dataset.

9. Unit normalization is allowed when the source explicitly
   provides convertible units.

10. If the document contains only one rack flow-pressure point,
    include that one point. Do not invent additional points.

11. If no multi-point or single-point rack flow-pressure data
    is explicitly available, return an empty array.

Possible document types include:

- rack datasheet
- server datasheet
- CDU datasheet
- coolant datasheet
- equipment specification
- bill of materials
- engineering specification

The output will be reviewed by an engineer
before any value is transferred into the H-LiquidOpt
deterministic calculation workflow.
"""

    try:
        response = client.responses.create(
            model=DEFAULT_MODEL,

            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": filename,
                            "file_data": (
                                f"data:application/pdf;base64,{encoded_file}"
                            ),
                        },
                        {
                            "type": "input_text",
                            "text": instructions,
                        },
                    ],
                }
            ],

            text={
                "format": {
                    "type": "json_schema",
                    "name": "hliquidopt_specification",
                    "strict": True,
                    "schema": SPEC_SCHEMA,
                }
            },
        )

    except openai.AuthenticationError as exc:
        raise RuntimeError(
            "OpenAI authentication failed. "
            "Check the API key configured in Streamlit Secrets."
        ) from exc

    except openai.RateLimitError as exc:
        raise RuntimeError(
            "The AI service is temporarily rate-limited "
            "or the API usage quota has been reached. "
            "Please try again later or check API billing."
        ) from exc

    except openai.APITimeoutError as exc:
        raise RuntimeError(
            "The AI analysis timed out. "
            "Try again or use a smaller PDF."
        ) from exc

    except openai.APIConnectionError as exc:
        raise RuntimeError(
            "Could not connect to the AI service. "
            "Please check the connection and try again."
        ) from exc

    except openai.BadRequestError as exc:
        raise RuntimeError(
            "The AI service could not process this PDF. "
            "The document may be unsupported, malformed, "
            "or contain content that cannot be processed."
        ) from exc

    except openai.APIStatusError as exc:
        raise RuntimeError(
            f"The AI service returned an unexpected error "
            f"(HTTP {exc.status_code}). Please try again."
        ) from exc

    if not response.output_text:
        raise RuntimeError(
            "AI returned an empty specification result."
        )

    try:
        result = json.loads(
            response.output_text
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "AI response could not be parsed as structured JSON."
        ) from exc

    return result
# =========================================================
# Final Design Cross-Check
# =========================================================

FINAL_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_status": {
            "type": "string",
            "enum": [
                "NO_OBVIOUS_CONFLICT",
                "REVIEW_REQUIRED",
                "INSUFFICIENT_DATA",
            ],
        },

        "summary": {
            "type": "string"
        },

        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string"
                    },
                    "status": {
                        "type": "string",
                        "enum": [
                            "CONSISTENT",
                            "REVIEW_REQUIRED",
                            "INSUFFICIENT_DATA",
                            "NOT_APPLICABLE",
                        ],
                    },
                    "message": {
                        "type": "string"
                    },
                    "source_fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "document_type",
                                "manufacturer",
                                "model",
                                "rated_power_kw",
                                "hcr",
                                "coolant_name",
                                "density_kg_m3",
                                "cp_kj_kgk",
                                "viscosity_mpas",
                                "property_temp_c",
                                "supply_temp_min_c",
                                "supply_temp_max_c",
                                "recommended_flow_lpm",
                                "pressure_drop_kpa",
                            ],
                        },
                    },
                },
                "required": [
                    "category",
                    "status",
                    "message",
                    "source_fields",
                ],
                "additionalProperties": False,
            },
        },

        "missing_verifications": {
            "type": "array",
            "items": {
                "type": "string"
            },
        },

        "next_actions": {
            "type": "array",
            "items": {
                "type": "string"
            },
        },
    },

    "required": [
        "overall_status",
        "summary",
        "checks",
        "missing_verifications",
        "next_actions",
    ],

    "additionalProperties": False,
}


def cross_check_design(
    verified_spec,
    final_decision,
    design_context,
    api_key=None,
):
    """
    AI-assisted final engineering cross-check.

    The function reviews a selected preliminary-design
    scenario against engineer-verified source constraints.

    It does NOT approve or certify the design.
    """

    client = get_openai_client(
        api_key
    )

    review_input = {
        "engineer_verified_specification": verified_spec,
        "engineer_selected_scenario": final_decision,
        "design_context": design_context,
    }

    instructions = """
You are the final engineering cross-check assistant
for H-LiquidOpt.

You are reviewing a preliminary D2C liquid-cooling design.

The engineering calculations have already been performed
by a deterministic physics engine.

Your job is NOT to recalculate the hydraulic model and
NOT to approve the design.

Review the engineer-selected scenario against the
engineer-verified source specification and identify:

- clear consistency
- possible conflicts
- missing evidence
- assumptions still requiring verification
- important next engineering checks

IMPORTANT RULES

1. Never certify, approve, or declare the design safe.

2. Never invent an OEM requirement.

3. Treat engineer_verified_specification as the verified
   source-reference dataset.

4. If information is insufficient, explicitly return
   INSUFFICIENT_DATA.

5. A documented recommended flow is NOT automatically
   a hard minimum or maximum unless the source explicitly
   established it as such.

6. Do NOT directly compare rack internal pressure drop
   with total system/network pressure drop.
   They represent different hydraulic boundaries.

7. A documented rack pressure drop may only be treated
   as a reference at its documented operating condition.

8. If the selected coolant differs from the documented
   coolant/formulation, do NOT automatically call it
   incompatible. Flag it for OEM/supplier review unless
   explicit incompatibility evidence exists.

9. Check the project supply temperature against the
   documented supply-temperature range when both exist.

10. Pipe-diameter sensitivity cases are conceptual
    preliminary-design scenarios, not final standard pipe
    size selections.

11. Pump power, pressure drop and flow calculated by
    H-LiquidOpt are deterministic calculation outputs.
    Do not replace them with invented values.

12. Highlight placeholders such as synthetic rack ΔP,
    missing OEM pressure-flow curves, material compatibility,
    water chemistry, fitting/minor losses, routing and CAPEX
    when relevant.

13. Keep the review concise and engineering-focused.

14. For every check, source_fields must list the canonical
    specification fields that directly support the finding.

15. Use only field names represented in the
    engineer-verified source traceability dataset.

16. If no source field directly supports a finding,
    return an empty source_fields array.

17. Never invent page numbers, quotations, source fields,
    or evidence.

18. Do not use AI review notes alone as evidence for a
    quantitative OEM requirement unless a corresponding
    verified source field exists.

19. If a quantitative requirement cannot be traced to a
    verified source field, classify it as REVIEW_REQUIRED
    or INSUFFICIENT_DATA rather than presenting it as fact.

Return only the required structured review.
"""

    response = client.responses.create(
        model="gpt-5.6-terra",

        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            instructions
                            + "\n\nDESIGN DATA:\n"
                            + json.dumps(
                                review_input,
                                ensure_ascii=False,
                                indent=2,
                            )
                        ),
                    }
                ],
            }
        ],

        text={
            "format": {
                "type": "json_schema",
                "name": "hliquidopt_final_crosscheck",
                "strict": True,
                "schema": FINAL_REVIEW_SCHEMA,
            }
        },

        store=False,
    )

    if not response.output_text:
        raise RuntimeError(
            "AI returned an empty final cross-check."
        )

    try:
        return json.loads(
            response.output_text
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "AI final cross-check could not be parsed."
        ) from exc
