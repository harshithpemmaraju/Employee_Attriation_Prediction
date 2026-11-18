# Employee Attrition Predictor (Streamlit + FastAPI)

Predicts whether an employee is **likely to leave** or **likely to stay**.
It is the deployment version of the notebook `employee_attriation_project.ipynb`.

- **Frontend:** Streamlit (`app.py`) - shows the form and the results.
- **Backend:** FastAPI (`main.py`) - loads `model.pkl`, checks the inputs, predicts and explains the prediction.

```
app.py (Streamlit)  --JSON-->  main.py (FastAPI)  -->  model.pkl
app.py (Streamlit)  <--JSON--  main.py (FastAPI)
```

## Project folder structure

```
Employee_Attrition_Streamlit/
├── app.py                  # Streamlit frontend (talks to the backend, never loads the model)
├── main.py                 # FastAPI backend (loads model.pkl, predicts)
├── train.py                # Trains the model and creates model.pkl
├── model.pkl               # Saved preprocessing + trained model (created by train.py)
├── requirements.txt        # Python libraries needed
├── README.md               # This file
├── .gitignore              # Keeps venv and cache files out of GitHub
└── employee_attrition.csv  # Original dataset (60,000 rows, 29 columns)
```

## What the model does

- **Target:** `Attrition` (1 = employee left, 0 = employee stayed). About 35% of employees left.
- **Inputs:** 28 features - 21 numeric and 7 categorical (Gender, Education, Department, JobRole,
  MaritalStatus, BusinessTravel, OverTime).
- **Preprocessing:** numeric columns -> median imputation + StandardScaler;
  categorical columns -> most-frequent imputation + OneHotEncoder (unseen categories are ignored safely).
- **Model:** Logistic Regression with the tuned settings from the notebook (`C=0.1`, `class_weight=None`, `max_iter=1000`).
- **Split:** 80% train / 20% test, `random_state=42`, stratified on `Attrition`.

### Results on the 20% hold-out test set (12,000 rows)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.8763 | 0.8320 | 0.8100 | 0.8208 | 0.9487 |

## API endpoints (main.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Check that the API is running |
| GET | `/schema` | Form details (columns, number ranges, dropdown options) |
| POST | `/predict` | Send `{"values": {...}}`, get class, probabilities and reasons |

Test page made automatically by FastAPI: http://127.0.0.1:8000/docs

## Installation (Windows + VS Code)

1. Install Python **3.11 or newer** from https://www.python.org/downloads/ (tick **"Add Python to PATH"**).
2. Open the project folder in VS Code (**File -> Open Folder**), then open a terminal (**Terminal -> New Terminal**).
3. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

## Run the project (two terminals, both in the project folder)

Terminal 1 - backend:
```
uvicorn main:app --reload
```

Terminal 2 - frontend:
```
streamlit run app.py
```

Your browser opens at http://localhost:8501. Keep both terminals running. Stop each with `Ctrl + C`.

## Train the model (only if needed)

```
python train.py
```
Run this if the backend says `model.pkl` cannot be loaded. Then restart the backend.
`requirements.txt` pins scikit-learn to 1.9.1 because `model.pkl` was saved with that version.
If you retrain with a different version, change that line to match.

## How to use the app

1. Go through the five tabs (Personal, Job, Experience, Pay, Satisfaction & Performance) and fill in the details.
2. Pick the **Department** first - the **Job Role** list then shows only roles from that department.
3. Click **Predict**.
4. Read the result: predicted class, probability of leaving / staying, and a graph of which inputs pushed the
   prediction toward Leave or Stay.
5. If something is inconsistent (for example Years At Company larger than Total Working Years), the backend
   sends a message and no prediction is made until it is fixed.

## Deploy online (backend on Render, frontend on Streamlit Community Cloud)

### Files that must be in the GitHub repository (all in the main folder, no sub-folders)

`main.py`, `app.py`, `model.pkl`, `train.py`, `employee_attrition.csv`, `requirements.txt`, `README.md`, `.gitignore`

Do NOT upload the `venv` folder (the `.gitignore` file keeps it out when you use Git).

### Step 1 - Put the project on GitHub

1. Create a free account at https://github.com and click **New repository** (for example `employee-attrition`).
2. Upload the files listed above (**Add file -> Upload files**, or use `git push`).

### Step 2 - Deploy the backend on Render

1. Create a free account at https://render.com (sign in with GitHub).
2. Click **New +** -> **Web Service** -> choose your repository.
3. Fill in:
   - **Language / Runtime:** Python 3
   - **Branch:** main
   - **Root Directory:** leave empty
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Open **Environment Variables** and add: `PYTHON_VERSION` = `3.12.3`
   (scikit-learn 1.9.1 needs Python 3.11 or newer.)
5. Click **Create Web Service** and wait until it says **Live**.
6. Copy the URL at the top (like `https://employee-attrition-xxxx.onrender.com`).
7. Test it in the browser:
   - `https://YOUR-URL/` shows a "running" message.
   - `https://YOUR-URL/docs` opens the test page. Open **POST /predict** -> **Try it out**, paste the example
     below and click **Execute**. You should get status 200 with a prediction.

```json
{
  "values": {
    "Age": 36,
    "Gender": "Female",
    "Education": "Bachelor",
    "Department": "Finance",
    "JobRole": "Accountant",
    "JobLevel": 2,
    "MaritalStatus": "Divorced",
    "BusinessTravel": "Non-Travel",
    "OverTime": "No",
    "TotalWorkingYears": 9.8,
    "YearsAtCompany": 4.8,
    "YearsInCurrentRole": 2.6,
    "YearsSinceLastPromotion": 2.4,
    "MonthlyIncome": 8658,
    "DistanceFromHome": 8,
    "JobSatisfaction": 3,
    "EnvironmentSatisfaction": 3,
    "WorkLifeBalance": 3,
    "RelationshipSatisfaction": 3,
    "TrainingTimesLastYear": 2,
    "PerformanceRating": 3,
    "PercentSalaryHike": 14,
    "StockOptionLevel": 1,
    "NumCompaniesWorked": 3,
    "PercentRemoteWork": 50,
    "DailyRate": 800,
    "HourlyRate": 65,
    "MonthlyRate": 14546
  }
}
```

### Step 3 - Deploy the frontend on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Choose your repository, branch `main`, and **Main file path:** `app.py`.
4. Click **Advanced settings**:
   - **Python version:** 3.12 (or any 3.11+)
   - **Secrets:** paste this one line (use YOUR Render URL, without `/docs` and without a trailing `/`):
     ```
     API_URL = "https://employee-attrition-xxxx.onrender.com"
     ```
5. Click **Deploy**. When it finishes you get your public app link.

### Step 4 - Update later

Change a file on GitHub (or `git push`). Render and Streamlit redeploy automatically.

### Troubleshooting

| Problem | What to do |
|---|---|
| Page says "Cannot get the form details from the backend" | Check the `API_URL` secret. Open `YOUR-URL/` in the browser. If it is slow, the free Render server is waking up - wait a minute and refresh. |
| First load takes up to a minute | Normal. Free Render servers sleep after about 15 minutes without use. |
| Render build fails on scikit-learn | `PYTHON_VERSION` is missing or too old. Set it to a 3.12.x version. |
| Render log says `model.pkl not found` | `model.pkl` is not in the main folder of the repository. |
| Render log says the model cannot be loaded | The scikit-learn version differs from the one that saved `model.pkl`. Run `python train.py` locally, upload the new `model.pkl`, and change the `scikit-learn==` line in `requirements.txt` to the version train.py prints. |
| Streamlit shows `ModuleNotFoundError` | A library is missing from `requirements.txt`. |

Both services install the same `requirements.txt`. That is slightly more than each needs but keeps the project simple.
The API has no password, so anyone with the URL can call `/predict`. This is fine for a learning project.

## Important notes

- The dataset appears to be clean/synthetic-like. Results on real company data may be quite different.
- The scores above come from a single train/test split. The model has **not** been validated on real employees and is
  **not** production-ready. Use it for learning and demonstration, not for HR decisions.
- Logistic Regression and Gradient Boosting scored almost the same in the notebook (F1 0.8208 vs 0.8189).
- Hyperparameter tuning in the notebook used `RandomizedSearchCV` with only 3 candidates per model.
- The "what influenced this prediction" graph describes how the model reacts to the inputs, not real-world cause and effect.
- Input ranges are limited to the ranges found in the training data, because the model should not be trusted outside them.
