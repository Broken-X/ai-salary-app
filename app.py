import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# ---------------- Core training logic (your Colab pipeline, wrapped) ---------------- #

def train_salary_model(df: pd.DataFrame, model_name: str = "XGBoost"):
    # --- Check target column ---
    if "salary_usd" not in df.columns:
        raise ValueError("Column 'salary_usd' not found in the uploaded CSV.")

    # --- Outlier removal on salary_usd ---
    original_rows = len(df)
    z_scores = np.abs(stats.zscore(df["salary_usd"]))
    threshold = 3
    df_cleaned = df[z_scores < threshold]
    cleaned_rows = len(df_cleaned)
    eliminated_entries = original_rows - cleaned_rows
    df = df_cleaned.copy()

    # --- Feature selection (same as your code) ---
    feature_columns = [
        "job_title", "experience_level", "employment_type", "company_location",
        "company_size", "employee_residence", "remote_ratio", "required_skills",
        "education_required", "years_experience", "industry", "company_name",
    ]

    missing_feats = [c for c in feature_columns if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing required columns in CSV: {missing_feats}")

    y = df["salary_usd"]
    X = df[feature_columns]

    # --- Train/validation split ---
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, random_state=1, test_size=0.2
    )

    # --- Preprocess: numeric + categorical ---
    cat_feats = X.select_dtypes(include=["object"]).columns
    num_feats = X.select_dtypes(exclude=["object"]).columns

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ]
    )

    # --- Choose model (no GridSearch to keep it fast) ---
    if model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=150,
            random_state=1,
            n_jobs=-1,
        )
    elif model_name == "ExtraTrees":
        model = ExtraTreesRegressor(
            n_estimators=150,
            random_state=1,
            n_jobs=-1,
        )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

    # --- Fit ---
    pipe.fit(train_X, train_y)

    # --- Predictions + metrics ---
    val_predictions = pipe.predict(val_X)

    val_mae = mean_absolute_error(val_y, val_predictions)
    val_r2 = r2_score(val_y, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(val_y, val_predictions))

    # --- Scatter plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=val_y, y=val_predictions, alpha=0.6, ax=ax)

    min_val = min(val_y.min(), val_predictions.min())
    max_val = max(val_y.max(), val_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], "--", linewidth=2, label="Perfect Prediction")

    ax.set_title(f"Actual vs Predicted Salary\n(MAE: ${val_mae:,.2f})")
    ax.set_xlabel("Actual Salary (USD)")
    ax.set_ylabel("Predicted Salary (USD)")
    ax.legend()
    ax.grid(True)

    summary = {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "eliminated_entries": eliminated_entries,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "val_rmse": val_rmse,
        "train_size": len(train_y),
        "test_size": len(val_y),
    }

    return summary, fig


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="AI Job Salary Model Trainer", layout="wide")
st.title("AI Job Salary Model Trainer")

st.write(
    """
    Upload your **ai_job_dataset.csv** (or compatible file) and train a model to predict salaries.
    The app will:
    - Remove outliers from `salary_usd`
    - Encode categorical features and scale numeric ones
    - Train the selected model
    - Show metrics and an Actual vs Predicted scatter plot
    """
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_name = st.selectbox(
    "Choose model",
    ["RandomForest", "ExtraTrees"],
    index=0,
)

if uploaded_file is not None:
    if st.button("Train model"):
        try:
            df = pd.read_csv(uploaded_file)
            summary, fig = train_salary_model(df, model_name)

            st.subheader("Metrics")
            st.write(
                f"""
                - **Model**: {model_name}  
                - **Original rows**: {summary['original_rows']}  
                - **Rows after outlier removal**: {summary['cleaned_rows']}  
                - **Eliminated entries**: {summary['eliminated_entries']}  

                - **Validation MAE**: ${summary['val_mae']:,.2f}  
                - **Validation R²**: {summary['val_r2']:.4f}  
                - **Validation RMSE**: ${summary['val_rmse']:,.2f}  

                - **Train size**: {summary['train_size']}  
                - **Test size**: {summary['test_size']}  
                """
            )

            st.subheader("Actual vs Predicted Salary")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Upload a CSV file to get started.")

