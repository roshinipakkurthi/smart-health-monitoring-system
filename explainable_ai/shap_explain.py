import pandas as pd
import joblib
import shap

# Load model and data
model = joblib.load('models/anomaly_model.pkl')
X = pd.read_csv('data/health_data.csv').drop('anomaly', axis=1)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot (opens a window or inline in notebook; in VS Code, use Python: Show Plot Viewer)
shap.summary_plot(shap_values, X)
