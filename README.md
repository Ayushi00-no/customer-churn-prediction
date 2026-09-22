
# Customer Churn Prediction

End-to-end machine learning project that predicts whether a telecom customer will churn, based on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Features
- Full EDA and preprocessing pipeline (pandas, seaborn)
- Trained and compared Logistic Regression, Random Forest, and XGBoost models
- Best model achieves ~80% accuracy / ROC-AUC ~0.84
- Served via a FastAPI REST API with a simple web UI for live predictions

## Tech Stack
Python, pandas, scikit-learn, XGBoost, FastAPI, HTML/JS

## Run Locally
\`\`\`bash
python -m venv venv
source venv/bin/activate   # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
\`\`\`
Then open http://127.0.0.1:8000/static/index.html

## Project Structure
- `notebooks/` — EDA and model training
- `app/` — FastAPI application
- `models/` — saved trained model + preprocessing objects
- `static/` — frontend

