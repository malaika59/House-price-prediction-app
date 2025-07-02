                                               California Housing Price Prediction (End-to-End Machine Learning Project)

This project is an end-to-end implementation of a machine learning workflow to predict median housing prices in California districts. It uses the California Housing Prices dataset and covers all essential steps including data fetching, exploration, preprocessing, model training, evaluation, and deployment using Streamlit.

Project Overview

The goal is to build a regression model that can estimate the median house value for a given California district based on several features like income, age of houses, total rooms, and location.

The project is implemented in Python using common machine learning libraries including pandas, scikit-learn, and joblib. The final model is deployed using Streamlit to make it accessible through a web interface.

Workflow Summary

1. Data Acquisition
- Dataset: California Housing Prices from the Hands-On Machine Learning book.
- Loaded using `fetch_california_housing()` or via the `housing.tgz` archive from GitHub.

2. Data Exploration
- Histograms for all features using Matplotlib.
- Scatter plots and correlation matrices to identify important features.
- Custom feature combinations like "rooms per household" created for better insight.

3. Data Preprocessing
- Handled missing values using `SimpleImputer`.
- Encoded categorical features with `OneHotEncoder`.
- Used `StandardScaler` to standardize numerical features.
- Combined steps using `ColumnTransformer` and `Pipeline` for efficiency and clarity.

4. Model Training
Trained and evaluated three regression models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Cross-validation was used for fair comparison, and Random Forest Regressor was selected as the final model due to better RMSE performance.

5. Model Evaluation
- Evaluated using RMSE and MAE metrics on both validation and test sets.
- Confidence interval estimated using the `scipy.stats.t.interval()` method.

6. Model Export
- Trained model and preprocessing pipeline saved using `joblib.dump()` with compression for portability.

 7. Deployment
- A web application was created using Streamlit.
- Users can input features like income, house age, population, etc.
- The app uses the saved model and pipeline to return a house value prediction.
- The app is deployed using [Streamlit Cloud](https://streamlit.io/cloud) with a simple interface.

Included Files

- `app.py` – Streamlit application script
- `my_model.pkl` – Trained and compressed Random Forest model
- `my_pipeline.pkl` – Preprocessing pipeline used to transform inputs
- `requirements.txt` – Python package dependencies
- `README.md` – Project documentation

