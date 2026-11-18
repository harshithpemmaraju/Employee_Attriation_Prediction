"""
app.py - FRONTEND (Streamlit)

This file only shows the form and the results.
The model and all the calculations are in the FastAPI backend (main.py).

Terminal 1 (backend):   uvicorn main:app --reload
Terminal 2 (frontend):  streamlit run app.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# Address of the backend.
#  - on your laptop it is http://127.0.0.1:8000
#  - on Streamlit Cloud, add  API_URL = "https://your-backend.onrender.com"  under Settings -> Secrets
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_URL = API_URL.rstrip("/")


# ---------------------------------------------------------------------------
# 1. SETTINGS: which fields go into which tab, and the help texts
# ---------------------------------------------------------------------------
FIELD_GROUPS = {
    " Personal": ["Age", "Gender", "MaritalStatus", "Education", "DistanceFromHome"],
    " Job": ["Department", "JobRole", "JobLevel", "BusinessTravel", "OverTime", "PercentRemoteWork"],
    " Experience": [
        "TotalWorkingYears", "NumCompaniesWorked", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "TrainingTimesLastYear",
    ],
    " Pay": ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate", "PercentSalaryHike", "StockOptionLevel"],
    " Satisfaction & Performance": [
        "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
        "RelationshipSatisfaction", "PerformanceRating",
    ],
}

HELP_TEXT = {
    "Age": "Employee age in years.",
    "DistanceFromHome": "Distance from home to work, as recorded in the dataset.",
    "JobLevel": "Job level, 1 to 5 as in the dataset.",
    "PercentRemoteWork": "Percentage of work done remotely (0-100).",
    "TotalWorkingYears": "Total years of work experience (all employers).",
    "YearsAtCompany": "Years at the current company. Cannot exceed total working years.",
    "YearsInCurrentRole": "Years in the current role. Cannot exceed years at the company.",
    "YearsSinceLastPromotion": "Years since last promotion. Cannot exceed years at the company.",
    "TrainingTimesLastYear": "Number of training sessions attended last year.",
    "MonthlyIncome": "Monthly income, in the units used by the dataset.",
    "PercentSalaryHike": "Last salary hike in percent.",
    "StockOptionLevel": "Stock option level, 0 to 3 as in the dataset.",
    "JobSatisfaction": "Rating on the dataset's 1-4 scale.",
    "EnvironmentSatisfaction": "Rating on the dataset's 1-4 scale.",
    "WorkLifeBalance": "Rating on the dataset's 1-4 scale.",
    "RelationshipSatisfaction": "Rating on the dataset's 1-4 scale.",
    "PerformanceRating": "Performance rating, 2 to 4 as in the dataset.",
}


def pretty(name):
    """'YearsAtCompany' -> 'Years At Company'"""
    result = ""
    for i, letter in enumerate(name):
        if i > 0 and letter.isupper() and not name[i - 1].isupper():
            result += " "
        result += letter
    return result


# ---------------------------------------------------------------------------
# 2. GET THE FORM DETAILS FROM THE BACKEND (cached, so it is not called on every click)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Contacting the backend (a free server can take up to a minute to wake up)...")
def get_schema():
    response = requests.get(f"{API_URL}/schema", timeout=90)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 3. INPUT WIDGETS
# ---------------------------------------------------------------------------
def number_field(col, info):
    """Slider or number box for one numeric column."""
    label = pretty(col)
    help_text = HELP_TEXT.get(col)
    key = f"in_{col}"

    if info["is_integer"]:
        smallest, largest, middle = int(info["min"]), int(info["max"]), int(round(info["median"]))
        if largest - smallest <= 10:                                  # small range -> slider
            return st.slider(label, smallest, largest, middle, step=1, help=help_text, key=key)
        return st.number_input(label, min_value=smallest, max_value=largest, value=middle,   # big range
                               step=1, help=help_text, key=key)

    # numbers with decimals
    return st.number_input(label, min_value=float(info["min"]), max_value=float(info["max"]),
                           value=float(round(info["median"], 1)), step=0.1, format="%.1f",
                           help=help_text, key=key)


def category_field(col, schema, values):
    """Dropdown for one text column."""
    options = schema["categories"][col]

    # Job roles depend on the department chosen before
    roles = schema["department_roles"]
    if col == "JobRole" and values["Department"] in roles:
        options = roles[values["Department"]]

    return st.selectbox(pretty(col), options, key=f"in_{col}", help=HELP_TEXT.get(col))


def draw_form(schema):
    """Draw the tabs and fields. Returns the entered values as a dictionary."""
    values = {}
    tabs = st.tabs(list(FIELD_GROUPS))

    for tab, fields in zip(tabs, FIELD_GROUPS.values()):
        with tab:
            left, right = st.columns(2)
            for i, col in enumerate(fields):
                box = left if i % 2 == 0 else right        # alternate left / right
                with box:
                    if col in schema["numeric_info"]:
                        values[col] = number_field(col, schema["numeric_info"][col])
                    else:
                        values[col] = category_field(col, schema, values)
    return values


# ---------------------------------------------------------------------------
# 4. GRAPH: why does the model say Leave or Stay?
# ---------------------------------------------------------------------------
def show_graph(reasons):
    labels = [r["label"] for r in reasons]
    effects = [r["effect"] for r in reasons]

    # red = pushes toward LEAVE, green = pushes toward STAY
    colors = ["#d9534f" if e > 0 else "#2e8b57" for e in effects]

    # [::-1] puts the strongest reason at the top
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.5), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(labels[::-1], effects[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Why the model predicts Leave or Stay (strongest factors)")
    ax.set_xlabel("<-- pushes toward STAY          pushes toward LEAVE -->")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption("Red bars push toward **Leave**, green bars push toward **Stay**. A longer bar means a "
               "stronger influence. The model adds up all these pushes: if the total favours Leave "
               "(probability above 50%) it predicts Leave, otherwise Stay.")

    leave_reasons = [r for r in reasons if r["effect"] > 0]
    stay_reasons = [r for r in reasons if r["effect"] < 0]
    if leave_reasons:
        best = max(leave_reasons, key=lambda r: r["effect"])
        st.write(f"**Strongest reason to LEAVE:** {best['label']}")
    if stay_reasons:
        best = min(stay_reasons, key=lambda r: r["effect"])
        st.write(f"**Strongest reason to STAY:** {best['label']}")


# ---------------------------------------------------------------------------
# 5. THE PAGE
# ---------------------------------------------------------------------------
def main():
    st.title(" Employee Attrition Predictor")
    st.markdown(
        "Predict whether an employee is **likely to leave** or **likely to stay**, based on the same "
        "28 employee features used to train the model. Fill in the tabs below and press **Predict**."
    )

    # --- get the form details from the backend ---
    try:
        schema = get_schema()
    except requests.exceptions.RequestException as error:
        st.error(f"Cannot get the form details from the backend at {API_URL}.\n\n"
                 "- On your laptop: start the backend with `uvicorn main:app --reload`.\n"
                 "- On Streamlit Cloud: check the `API_URL` secret and that the Render service is running.\n\n"
                 f"Details: `{error}`")
        st.stop()

    # --- form ---
    st.subheader("Employee details")
    values = draw_form(schema)

    st.divider()
    if not st.button(" Predict", type="primary"):
        return

    # --- send the values to the backend ---
    try:
        with st.spinner("Predicting..."):
            response = requests.post(f"{API_URL}/predict", json={"values": values}, timeout=90)
    except requests.exceptions.RequestException:
        st.error(f"Cannot connect to the backend at {API_URL}. Is it running? (A free Render server may be waking up - try again in a minute.)")
        return

    if response.status_code == 422:                       # the inputs were not valid
        st.error("Please fix the following before predicting:")
        for problem in response.json()["detail"]["problems"]:
            st.write(f"- {problem}")
        return
    if response.status_code != 200:
        st.error(f"The backend returned an error: {response.text}")
        return
    result = response.json()

    # --- show the result ---
    st.subheader("Result")
    if result["predicted_class"] == 1:
        st.error("### ⚠️ Likely to LEAVE")
    else:
        st.success("### ✅ Likely to STAY")

    p_leave = result["probability_leave"]
    p_stay = result["probability_stay"]
    left_col, right_col = st.columns(2)
    left_col.metric("Probability of leaving", f"{p_leave:.1%}")
    right_col.metric("Probability of staying", f"{p_stay:.1%}")
    st.progress(min(max(p_leave, 0.0), 1.0), text=f"Attrition risk: {p_leave:.1%}")

    # --- graph with the reasons ---
    if result["reasons"]:
        st.subheader("What influenced this prediction most?")
        show_graph(result["reasons"])

    with st.expander("Show the exact input row sent to the model"):
        st.dataframe(pd.DataFrame([result["input_row"]]))


main()
