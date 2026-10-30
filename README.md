# LoanLens — Loan Approval Prediction System (Pro)

A responsive educational ML mini-project built with Python, Flask, Pandas and Scikit-learn.

## What's included
- Loan category chooser: Home Loans, Business Loans, Personal Loans.
- Home loan types: Home Purchase, Construction, Extension, Renovation, Plot + Construction, Top-Up, Balance Transfer, Resale Property, Under-Construction Property and Joint Home Loan.
- Business loan types: MSME, Working Capital, Term Loan, GST Loan, Business Expansion, Machinery, OD/CC, Loan Against Property, Commercial Property, Invoice Financing, Commercial Vehicle, Shop/Office, Startup and Business Top-Up/Balance Transfer.
- Personal loan types: General Personal, Medical Emergency, Wedding, Travel, Education, Debt Consolidation, Balance Transfer and Top-Up.
- Loan overview, common documents and common eligibility considerations for each loan type.
- Applicant prediction form with input validation.
- Logistic Regression, Decision Tree and Random Forest comparison.
- Accuracy, precision, recall, F1-score and confusion matrix; best model is selected by F1-score.
- Responsive desktop/mobile styling.

## Important limitation
The included CSV is synthetic demonstration data. The model is based on generic sample features and is **not separately trained for every loan category or loan type**. Category/type selection changes the displayed information and is recorded with the result, but does not change the model's learned rules. Predictions are not bank decisions, credit advice, or guarantees. Do not use for real lending decisions.

## Python version
Use **Python 3.14 or 3.13** for this updated package list. Avoid Python 3.15 for now because some scientific packages may not yet provide compatible prebuilt wheels. If your PC only has Python 3.14, commands below use `py -3.14`.

## Windows / VS Code setup
1. Extract this ZIP.
2. Open the extracted `Loan_Approval_Prediction_System_Pro` folder in VS Code.
3. Open Terminal → New Terminal.
4. Check installed Python versions:
   ```powershell
   py -0p
   ```
5. Create a fresh virtual environment with Python 3.14:
   ```powershell
   py -3.14 -m venv .venv
   ```
   If you have Python 3.13 instead, use `py -3.13 -m venv .venv`.
6. Upgrade pip and install dependencies:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
7. Run the site:
   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```
8. Open `http://127.0.0.1:5000` in Chrome.

If a virtual environment created with a different Python version already exists, use a new name such as `.venv314` rather than reusing it:
```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv314\Scripts\python.exe app.py
```

## Project files
- `app.py` — Flask routes, training pipeline, prediction and evaluation.
- `loan_prediction_dataset.csv` — synthetic dataset for the demo.
- `templates/index.html` — page, loan category/type interaction and model cards.
- `static/style.css` — responsive visual design.
- `requirements.txt` — Python dependencies.
- `SMOKE_TEST.json` — sample project test notes.

## ML pipeline
1. Read the CSV and separate features from the target label.
2. Split into training and testing data (80/20, stratified).
3. Impute missing values, one-hot encode categorical features and scale numeric features.
4. Train three classifiers and calculate metrics on the test split.
5. Select the model with the highest F1-score for the demo prediction.

## Troubleshooting
- `vswhere.exe` or pandas build errors: confirm you are using Python 3.14/3.13 and a fresh environment, not Python 3.15.
- `py -3.14` not found: install Python 3.14 (64-bit) from python.org or use a version shown by `py -0p`.
- Port already in use: stop the other Flask process or change the port at the bottom of `app.py`.
- First start trains the models and may take a short time.
