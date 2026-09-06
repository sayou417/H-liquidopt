# GitHub Upload Guide

## 1. Unzip and enter the project folder

```bash
cd H-LiquidOpt_GitHub_Prototype
```

## 2. Test locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pytest -q
streamlit run app.py
```

Expected test result for the final package: **5 passed**.

## 3. Create a GitHub repository

Create an empty repository named `H-LiquidOpt` on GitHub, then run:

```bash
git init
git add .
git commit -m "Initial H-LiquidOpt prototype"
git branch -M main
git remote add origin https://github.com/YOUR_ID/H-LiquidOpt.git
git push -u origin main
```

## 4. Contest demo

1. Start with the included 48-rack sample.
2. Approve Phase 1 after reviewing rack power/HCR and the heat-load map.
3. Select a TCS/CDU candidate in Phase 2.
4. Select numerical coolant sensitivity candidates in Phase 3.
5. Review deterministic hydraulic results in Phase 4.
6. Show the candidate comparison and download the design-review report in Phase 5.
7. Upload a different-size rack CSV to demonstrate that the engine is not limited to 48 racks.

## Important

The prototype is for preliminary design demonstration only. Replace synthetic geometry, rack ΔP, coolant properties and other placeholders with validated project/OEM/supplier data before engineering use.
