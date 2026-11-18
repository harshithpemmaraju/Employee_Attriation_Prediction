"""
main.py - BACKEND (FastAPI)

The backend does the "thinking": it loads the model, checks the inputs,
predicts, and finds the reasons behind the prediction.
The Streamlit app (app.py) is the FRONTEND: it only shows the form and results.

Start the backend:  uvicorn main:app --reload
Test page (auto):   http://127.0.0.1:8000/docs
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. LOAD THE MODEL (one time, when the server starts)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

if not MODEL_PATH.exists():
    raise RuntimeError("model.pkl not found. Run `python train.py` first.")

bundle = joblib.load(MODEL_PATH)      # a dictionary saved by train.py
pipeline = bundle["pipeline"]         # preprocessing + model together

app = FastAPI(title="Employee Attrition API")


# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------
def pretty(name):
    """'YearsAtCompany' -> 'Years At Company'"""
    result = ""
    for i, letter in enumerate(name):
        if i > 0 and letter.isupper() and not name[i - 1].isupper():
            result += " "
        result += letter
    return result


def validate(values):
    """Check the inputs. Returns a list of problems (empty list = all OK)."""
    problems = []

    # a) every field must be filled (and numeric fields must really be numbers)
    for col in bundle["feature_columns"]:
        value = values.get(col)
        if value is None or value == "":
            problems.append(f"'{pretty(col)}' is missing.")
        elif col in bundle["numeric_info"] and (isinstance(value, bool) or not isinstance(value, (int, float))):
            problems.append(f"'{pretty(col)}' must be a number.")
    if problems:
        return problems

    # b) numbers must be inside the range seen in training
    for col, info in bundle["numeric_info"].items():
        if not (info["min"] <= values[col] <= info["max"]):
            problems.append(f"{pretty(col)} must be between {info['min']:g} and {info['max']:g}.")

    # c) dropdown values must be known categories
    for col, allowed in bundle["categories"].items():
        if values[col] not in allowed:
            problems.append(f"'{values[col]}' is not a known value for {pretty(col)}.")

    # d) rules between fields
    if values["YearsAtCompany"] > values["TotalWorkingYears"]:
        problems.append("Years At Company cannot be greater than Total Working Years.")
    if values["YearsInCurrentRole"] > values["YearsAtCompany"]:
        problems.append("Years In Current Role cannot be greater than Years At Company.")
    if values["YearsSinceLastPromotion"] > values["YearsAtCompany"]:
        problems.append("Years Since Last Promotion cannot be greater than Years At Company.")
    if values["TotalWorkingYears"] > values["Age"] - 14:
        problems.append("Total Working Years looks too high for this Age (the data has at least 14 years "
                        "between age and years worked).")

    # e) the job role must belong to the chosen department
    roles = bundle.get("department_roles", {})
    if values["Department"] in roles and values["JobRole"] not in roles[values["Department"]]:
        problems.append(f"Job Role '{values['JobRole']}' does not belong to the "
                        f"'{values['Department']}' department.")

    return problems


def find_reasons(values, row, top_n=10):
    """
    Which inputs pushed the prediction toward Leave (+) or Stay (-)?
    reason = model weight x (scaled / encoded input value)
    Works for Logistic Regression. Returns [] for other models.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "coef_"):
        return []

    # same preprocessing as in training
    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    # one number for every transformed column
    numbers = transformed[0] * model.coef_[0]
    names = preprocessor.get_feature_names_out()

    # add the one-hot columns (Department_Sales, Department_IT ...) back to "Department"
    totals = {}
    for name, number in zip(names, numbers):
        name = name.split("__", 1)[1]                 # remove "num__" or "cat__"
        for col in bundle["feature_columns"]:
            if name == col or name.startswith(col + "_"):
                totals[col] = totals.get(col, 0) + float(number)
                break

    # keep the strongest ones
    strongest = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]

    reasons = []
    for col, effect in strongest:
        value = values[col]
        if isinstance(value, (int, float)):
            value = f"{value:g}"
        reasons.append({"label": f"{pretty(col)} = {value}", "effect": effect})
    return reasons


# ---------------------------------------------------------------------------
# 3. ENDPOINTS (the addresses the frontend can call)
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    values: dict          # example: {"Age": 36, "Gender": "Male", ...}


@app.get("/")
def home():
    return {"message": "Employee Attrition API is running. Open /docs to test it."}


@app.get("/schema")
def schema():
    """Details the frontend needs to build the form."""
    numeric_info = {}
    for col, info in bundle["numeric_info"].items():
        numeric_info[col] = {
            "min": float(info["min"]),
            "max": float(info["max"]),
            "median": float(info["median"]),
            "is_integer": bool(info["is_integer"]),
        }
    return {
        "feature_columns": list(bundle["feature_columns"]),
        "numeric_info": numeric_info,
        "categories": {col: [str(v) for v in options] for col, options in bundle["categories"].items()},
        "department_roles": bundle.get("department_roles", {}),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    values = request.values

    # 1. check the inputs
    problems = validate(values)
    if problems:
        raise HTTPException(status_code=422, detail={"problems": problems})

    # 2. one-row table, columns in the same order as in training
    row = pd.DataFrame([values], columns=bundle["feature_columns"])

    # 3. predict
    predicted_class = int(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0]
    classes = list(pipeline.classes_)

    # 4. send the answer back
    return {
        "predicted_class": predicted_class,                       # 1 = Leave, 0 = Stay
        "probability_leave": float(probabilities[classes.index(1)]),
        "probability_stay": float(probabilities[classes.index(0)]),
        "reasons": find_reasons(values, row),
        "input_row": {col: values[col] for col in bundle["feature_columns"]},
    }
