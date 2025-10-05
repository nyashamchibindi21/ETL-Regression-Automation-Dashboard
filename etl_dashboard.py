pip install seaborn



import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import shap

# Streamlit UI
st.title("ETL Regression Automation Dashboard")
st.write("""
This app automates the ETL (Extract, Transform, Load) process for regression analysis.
Ensure your dataset is preprocessed (e.g., handle missing values, encode categorical variables) before uploading.
""")

# File Upload
uploaded_file = st.file_uploader("Upload your preprocessed CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Select Target Variable (y)
    st.write("### Step 1: Select Target Variable")
    y_column = st.selectbox("Choose the target variable (y)", df.columns)
    st.write(f"You selected `{y_column}` as the target variable.")

    # Select Features (X)
    st.write("### Step 2: Select Features")
    X_columns = st.multiselect("Choose the feature variables (X)", [col for col in df.columns if col != y_column])
    st.write(f"You selected the following features: `{X_columns}`")

    if y_column and X_columns:
        X = df[X_columns]
        y = df[y_column]

        # Train-Test Split
        st.write("### Step 3: Train-Test Split")
        test_size = st.slider("Select the test size ratio (e.g., 0.2 for 20%)", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write(f"Training set size: {X_train.shape[0]} samples")
        st.write(f"Test set size: {X_test.shape[0]} samples")

        # Data Normalization
        st.write("### Step 4: Data Normalization")
        normalize_data = st.radio("Do you want to normalize the data?", ("Yes", "No"))
        
        if normalize_data == "Yes":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.write("Data has been normalized.")
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            st.write("Data normalization skipped.")

        # Regression Models
        st.write("### Step 5: Regression Analysis")
        st.write("Select the regression models you want to train:")
        
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "SVR": SVR(),
            "KNN Regressor": KNeighborsRegressor(),
            "XGBoost Regressor": XGBRegressor()
        }
        
        selected_models = st.multiselect("Choose models to train", list(models.keys()))
        
        if selected_models:
            model_results = {}
            for name in selected_models:
                model = models[name]
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                model_results[name] = score

            # Display Model Performance
            st.write("### Model Performance")
            results_df = pd.DataFrame(list(model_results.items()), columns=["Model", "R2 Score"])
            results_df = results_df.sort_values(by="R2 Score", ascending=False)
            st.dataframe(results_df)

            # Best Model Download
            st.write("### Step 6: Download the Best Model")
            best_model_name = results_df.iloc[0, 0]
            best_model = models[best_model_name]
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            st.download_button(
                label="Download Best Model",
                data=open("best_model.pkl", "rb").read(),
                file_name="best_model.pkl",
                mime="application/octet-stream"
            )
            st.write(f"The best model is `{best_model_name}` with an R² score of {results_df.iloc[0, 1]:.2f}.")

            # Feature Importance
            st.write("### Step 7: Feature Importance")
            best_model.fit(X_train, y_train)
            if hasattr(best_model, "feature_importances_"):
                feature_importance = best_model.feature_importances_
            else:
                explainer = shap.Explainer(best_model, X_train)
                shap_values = explainer(X_train)
                feature_importance = np.abs(shap_values.values).mean(axis=0)
            
            feature_importance_df = pd.DataFrame({"Feature": X_columns, "Importance": feature_importance})
            feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(x=feature_importance_df["Importance"], y=feature_importance_df["Feature"], ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)

# Preprocessing Warning
st.sidebar.warning("""
**Important Note:**  
Ensure your dataset is preprocessed before uploading.  
- Handle missing values.  
- Encode categorical variables.  
- Remove unnecessary columns.  

""")
