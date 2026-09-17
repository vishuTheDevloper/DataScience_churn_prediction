# Customer Churn Prediction

This project predicts telecom customer churn using a Decision Tree Classifier. It includes an analysis notebook, reusable preprocessing, a saved model pipeline, and a FastAPI `POST /predict` endpoint.

**To try predictions, follow sections 1–4. Retraining is optional when the saved pipeline is included.**

# Git hub public repo link below 
https://github.com/vishuTheDevloper/DataScience_churn_prediction

This project predicts telecom customer churn using a Decision Tree Classifier. It includes an analysis notebook, reusable preprocessing, a saved model pipeline, and a FastAPI `POST /predict` endpoint.
## 1. Requirements and ZIP contents

The project was developed on Windows with Python 3.13.15. Use Python 3.13 and the package versions supplied in `requirements.txt` to reproduce the environment. Internet access is needed to install dependencies; inference uses the local saved model and requires no API key, cloud account, or GPU.

Install Python from [the official Python website](https://www.python.org/downloads/). VS Code with the Microsoft Python and Jupyter extensions is needed only for the notebook instructions below; it is not needed to run the API from a terminal.

Extract the ZIP fully before running any commands. The project root is the extracted folder containing `README.md`, `requirements.txt`, and `src/`. Keep these files and directories together:

| Path | Purpose |
|---|---|
| `README.md` | Setup and execution instructions |
| `requirements.txt` | Package versions from the working project environment |
| `data/TelcoCustomerChurn.csv` | Dataset for analysis and retraining |
| `data/TelcoCustomerChurn - Data Dictionary.csv` | Dataset field descriptions |
| `notebooks/customer_churn.ipynb` | Analysis, training, evaluation, and explanations |
| `src/preprocessing.py` | Custom preprocessing transformer used by the saved pipeline |
| `src/schemas.py` | Customer input rules and prediction response schema |
| `src/api.py` | FastAPI application |
| `models/churn_pipeline_tuned.joblib` | Selected model, fitted imputers, and categorical encoder |
| `examples/sample_request.json` | Complete example customer input |
| `examples/sample_response.json` | Actual response recorded for that input |

The API uses `churn_pipeline_tuned.joblib`. An older `churn_pipeline.joblib`, if present, is not used by this API.

Create your own virtual environment after extraction. A virtual environment copied from someone else's computer is not portable.

## 2. Create the environment and install packages

### Windows — Command Prompt

Open the extracted project folder in File Explorer. Click its address bar, type `cmd`, and press Enter. This opens Command Prompt at the project root, without requiring the sender's username or folder path.

Run these commands one at a time:

```bat
py -3.13 --version
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip check
```

The prompt should show `(.venv)`. A successful dependency check reports `No broken requirements found.`

If `py` is unavailable but `python --version` reports Python 3.13, use `python -m venv .venv` for environment creation instead. If neither command works, install Python 3.13 and reopen Command Prompt.

These Windows commands are for **Command Prompt**, not PowerShell. In VS Code, choose **Command Prompt** from the terminal profile menu when following them.

### macOS / Linux — alternative setup

The original setup was verified on Windows. On another operating system, install Python 3.13, open a terminal in the extracted project root, and use:But not tested on Mac

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

If a pinned dependency is unavailable on your platform, use the original Windows environment or resolve the platform compatibility issue before proceeding. Do not silently change scikit-learn versions and assume the saved model remains compatible.

## 3. Start the prediction API

With the environment active, run this from the project root:

```bash
python -m uvicorn src.api:app --reload
```

Wait for `Application startup complete.` The application loads the saved pipeline once at startup and reuses it for predictions. It does not train a model while handling requests.

Open the interactive documentation in your browser:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Keep the terminal open while using the API. Press **Ctrl+C** in that terminal to stop it. The command runs locally on your computer; `--reload` is for local development and testing.

## 4. Send a sample prediction request

### Through the browser

1. Open `/docs` at the URL above.
2. Expand **POST /predict**.
3. Click **Try it out**.
4. Replace the request body with the following JSON, also provided in `examples/sample_request.json`.
5. Click **Execute** and inspect **Server response**.

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 80.5,
  "TotalCharges": 402.5
}
```

### Through a second terminal

Leave the API running in the first terminal. Open a second terminal in the project root.

Windows Command Prompt:

```bat
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@examples/sample_request.json"
```

macOS / Linux:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@examples/sample_request.json"
```

### Recorded successful response

The example returned HTTP `200` with the following body in the original environment:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.7142857142857143
}
```

This is also saved in `examples/sample_response.json`.

- `prediction` is `Yes` for predicted churn or `No` for predicted non-churn.
- `churn_probability` is the score for the **Yes class**, even when the predicted class is No. It is not the maximum probability across both classes.
- The tree uses class weights and has not been probability-calibrated. Interpret the value as the model's churn score, not a verified individual risk.

## 5. Input validation and missing values

Send all 19 original customer fields using the exact names and category spellings in the sample. Do not include `customerID`, `Churn`, or engineered features; unexpected fields are rejected.

| Input | Accepted values |
|---|---|
| `gender` | `Female`, `Male` |
| `SeniorCitizen` | Integer `0` or `1` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | `Yes`, `No` |
| `tenure` | Non-negative integer, in months |
| `MultipleLines` | `Yes`, `No`, `No phone service` |
| `InternetService` | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | `Yes`, `No`, `No internet service` |
| `Contract` | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | Non-negative finite number |
| `TotalCharges` | Non-negative finite number, or JSON `null` when unknown |

Use JSON numbers rather than numeric strings. For example, use `"tenure": 5`, not `"tenure": "5"`. Other than `TotalCharges`, missing or null customer fields are rejected by the API schema.

Service combinations must be consistent:

- When `PhoneService` is `No`, `MultipleLines` must be `No phone service`; otherwise it must be `Yes` or `No`.
- When `InternetService` is `No`, all six internet add-on fields must be `No internet service`; otherwise they must be `Yes` or `No`.

For `"TotalCharges": null`, the saved pipeline fills the missing value using the median learned from training data. It does not compute a new median from the request.

### Validation checks

The following cases were manually checked through the API. Start from the original valid request for each case.

| Change | Expected behavior | Observed behavior |
|---|---|---|
| No change | HTTP 200 with prediction and churn score | Passed |
| Set `tenure` to `-5` | HTTP 422 explaining the minimum value | Rejected as expected |
| Set `PhoneService` to `No`, leaving `MultipleLines` as `No` | HTTP 422 explaining the service conflict | Rejected as expected |
| Set `TotalCharges` to `null` | HTTP 200 after pipeline imputation | Passed |

## 6. Run the notebook and retrain — optional

You can run the included saved model without running the notebook first. To inspect the analysis or regenerate the model:

1. Stop the API with **Ctrl+C** before replacing its model file.
2. Open the extracted project root in VS Code.
3. Install the Microsoft **Python** and **Jupyter** extensions if needed.
4. Confirm `data/TelcoCustomerChurn.csv` and `src/preprocessing.py` are present.
5. Open `notebooks/customer_churn.ipynb`.
6. Use **Select Kernel** to select the project's `.venv` Python environment. On Windows its executable is `.venv\Scripts\python.exe`.
7. Restart the kernel, then run all cells in order. The grid search can take several minutes depending on the computer.
8. Resolve any cell errors before continuing, and save the notebook with its outputs.
9. Confirm the final saving cell reports `Reload check passed: predictions and probabilities match.`

The notebook writes `models/churn_pipeline_tuned.joblib`, replacing that file if it already exists. Copy the existing model elsewhere first if you want to preserve the exact supplied artifact. Restart the API after retraining to load the newly saved model.

## 7. Analysis and model summary

The data contains 7,043 rows and 21 columns. `customerID` is excluded and `Churn` is encoded as No = 0 and Yes = 1. Eleven whitespace entries in `TotalCharges` are treated as missing; no duplicate rows or customer IDs were found.

The notebook includes structure and missing-value checks, five main EDA visualizations, and two engineered features: `TenureGroup` and `ServiceCount`. It compares models with and without these features.

Training and testing use a stratified **70:30 split** with **random_state = 42**: 4,930 training rows and 2,113 test rows. Numerical imputation, categorical imputation, and one-hot encoding are inside the pipeline. Learned preprocessing is fitted separately within each training fold during cross-validation.

Model selection compares unrestricted and constrained decision trees, class weights, and a grid search using five-fold stratified cross-validation. Mean validation F1 is the selection criterion; precision and recall are also inspected.

Selected settings:

| Setting | Value |
|---|---|
| Classifier | Decision Tree |
| Criterion | `gini` |
| Maximum depth | `5` |
| Minimum samples per leaf | `40` |
| Class weights | `{0: 1, 1: 2}` |
| Random state | `42` |
| Engineered features in selected model | Disabled |

### Evaluation

| Metric | Mean cross-validation score | Reused test-set score |
|---|---:|---:|
| Accuracy | 0.770 | 0.762 |
| Precision | 0.552 | 0.539 |
| Recall | 0.717 | 0.720 |
| F1 | 0.624 | 0.616 |

Test confusion matrix:

| Actual outcome | Predicted stayed | Predicted churned |
|---|---:|---:|
| Stayed | 1206 | 346 |
| Churned | 157 | 404 |

The selected model identifies 404 of 561 actual churners, with 346 false alerts. Compared with the initial balanced tree, precision and F1 improve while recall decreases. Contract type, tenure, and internet service have the largest feature importances in the selected tree. These are associations, not proof of causation.

**Evaluation limitation:** The initial model's test results were viewed before follow-up development. Later tuning and model selection used training-data cross-validation, but the final test report reuses the previously viewed test set. It is not a fresh independent holdout evaluation.

## 8. Troubleshooting

| Problem | What to do |
|---|---|
| PowerShell blocks `Activate.ps1` | Use Command Prompt and `.venv\Scripts\activate.bat`. |
| `No module named uvicorn` or another missing package | Activate the project's environment and run `python -m pip install -r requirements.txt`. |
| `No module named src` | Run the Uvicorn command from the project root and confirm the complete `src` folder was extracted. |
| Saved pipeline not found | Include `models/churn_pipeline_tuned.joblib`, or run the notebook to generate it. |
| Model-loading version warning/error | Recreate the environment using the supplied requirements and original Python version. Loading across different scikit-learn versions is unsupported. If necessary, retrain in a compatible environment. |
| Browser cannot connect | Check the terminal for startup errors and leave the server running. Use the same port shown in the terminal. |
| Opening `/` gives Not Found | Open `/docs`; a homepage route is not defined. |
| Opening `/predict` in the address bar gives Method Not Allowed | Use POST through `/docs` or the supplied curl command. A browser address-bar request uses GET. |
| Port 8000 is occupied | Run `python -m uvicorn src.api:app --reload --port 8001`, then use port 8001 in browser and request URLs. |
| Notebook cannot import installed packages | Select the project's `.venv` kernel and restart it. |
| API returns HTTP 422 | Read the response's `detail` field and correct the named input or conflicting service values. |

## 9. Sender: prepare the ZIP

Before sharing, save the notebook with its outputs, include the selected model and all files listed in section 1, and refresh dependencies from the same environment used to train and serve the model.

In a separate **Command Prompt** at the project root:

```bat
.venv\Scripts\activate.bat
python -m pip freeze > requirements.txt
python -m pip check
```

Open `requirements.txt` and confirm it contains package names and versions rather than being empty. Do not replace it with package versions from a different computer.

For the ZIP, select `data`, `notebooks`, `src`, `models`, `examples`, `requirements.txt`, and `README.md` in File Explorer, then use **Compress to ZIP file**. Include any additional required assignment deliverables as applicable.

Exclude `.venv` and `.venv_old` from the archive. Keep your working `.venv` on your own computer; the recipient creates a new one using section 2.

