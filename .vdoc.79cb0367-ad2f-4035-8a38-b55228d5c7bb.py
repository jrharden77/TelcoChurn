# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: setup_and_load
#| include: true 

# ------------------------------------------------------------------
# 1. SETUP & LIBRARIES
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tabulate import tabulate
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Set style for charts
sns.set_style("whitegrid")

# ------------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------------
# Note: Ensure this path is correct for your local environment
data_folder = r'G:\Other computers\My laptop\Google Drive\Resume\GitHub\TelcoChurn\.gitignore'

# Added skipinitialspace=True to handle potential spaces after commas in CSVs
df = pd.read_csv(os.path.join(data_folder, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'), skipinitialspace=True)
#
#
#
#| label: label_encoding
#| include: false

# --- Total Column Head ---
#['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
#       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
#       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
#       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

# --- Things to Encode ---
# Gender = ['Female', 'Male']
# Partner = ['Yes', 'No']
# Dependents = ['Yes', 'No']
# PhoneService = ['Yes', 'No']
# MultipleLines = ['No phone service', 'No', 'Yes']
# InternetService = ['DSL', 'Fiber optic', 'No']
# OnlineSecurity = ['Yes', 'No']
# DeviceProtection = ['Yes', 'No']
# TechSupport = ['Yes', 'No']
# StreamingTV = ['Yes', 'No']
# StreamingMovies = ['Yes', 'No']
# Contract = ['Month-to-month', 'One year', 'Two year']
# PaperlessBilling = ['Yes', 'No']
# PaymentMethod = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
# Churn = ['Yes', 'No']

# --- Encoding all Yes/No as 1/0 ---
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0})
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0})
df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0})
df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0})
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# --- Specific Encoding ---
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 2, 'Yes': 1, 'No': 0})
df['InternetService'] = df['InternetService'].map({'DSL':2, 'Fiber optic': 1, 'No': 0})
df['OnlineBackup'] = df['OnlineBackup'].map({'No internet service':2, 'Yes': 1, 'No': 0})
df['Contract'] = df['Contract'].map({'Two Year': 2, 'One year': 1, 'Month-to-month': 0})
df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check':3, 'Mailed check':2, 'Bank transfer (automatic)': 1, 'Credit card (automatic)': 0})

# --- Handling TotalCharges Values ---
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
#
#
#
#| label: train_test_split
#| include: false

X = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]

y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=786)
#
#
#
#| label: train_model
#| include: false

# 1. Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=786)

# 2. Train the model (Fit)
# Note: Ensure you use lowercase x_train (matches your previous block)
rf_model.fit(X_train, y_train)
#
#
#
#| label: interpret_results
#| include: false

y_proba = rf_model.predict_proba(X_test)[:, 1]

CUSTOM_THRESHOLD = 0.3
y_pred = (y_proba >= CUSTOM_THRESHOLD).astype(int)

# --- Confusion Matrix ---
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title(f"Confusion Matrix (Threshold: {CUSTOM_THRESHOLD})")
plt.show()

# --- ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# --- Plots ---
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
#
#
#
#| label: feature_importance_plot
#| include: false

# 1. Get the numbers (Feature Importances)
importances = rf_model.feature_importances_
feature_names = X_train.columns

# 2. Create a clean DataFrame
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# 3. Calculate "Direction" (Correlation) to see if it's Good or Bad
# Note: We need to combine X and y temporarily to calculate this
combined_data = X_train.copy()
combined_data['Churn'] = y_train
correlations = combined_data.corrwith(combined_data['Churn'])

# Map the correlations to our importance dataframe
feature_imp_df['Correlation'] = feature_imp_df['Feature'].map(correlations)
feature_imp_df['Impact'] = np.where(feature_imp_df['Correlation'] > 0, 'Increases Churn (Bad)', 'Decreases Churn (Good)')

# 4. Sort by Importance (Highest at top)
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# 5. Plot
plt.figure(figsize=(10, 8))
sns.barplot(
    data=feature_imp_df.head(10), # Top 10 only
    x='Importance', 
    y='Feature', 
    hue='Impact', # Color by Good/Bad
    palette={'Increases Churn (Bad)': '#d62728', 'Decreases Churn (Good)': '#2ca02c'},
    dodge=False
)

plt.title('Top 10 Drivers of Churn')
plt.xlabel('Importance (Weight)')
plt.ylabel('Feature')
plt.legend(title='Impact Direction', loc='lower right')
plt.tight_layout()
plt.show()
#
#
#
#| label: export_hit_list
#| include: false

# 1. Select Active Customers (Churn == 0)
active_customers = df[df['Churn'] == 0].copy()

# 2. Calculate Risk
# Get the probability of '1' (Churn)
active_customers['Churn_Risk'] = rf_model.predict_proba(X[df['Churn'] == 0])[:, 1]

# 3. Calculate "Expected Loss" (Risk * Money)
active_customers['Expected_Loss'] = active_customers['Churn_Risk'] * active_customers['MonthlyCharges']

# 4. Create the "Hit List" (Top 500 Riskiest Dollars)
hit_list = active_customers[['customerID', 'Churn_Risk', 'MonthlyCharges', 'Expected_Loss', 'Contract', 'tenure']]
hit_list = hit_list.sort_values(by='Expected_Loss', ascending=False).head(500)

total_expected_loss = hit_list['Expected_Loss'].sum()

# Format Percentage (e.g., 0.658 -> 65.8%)
# .1% means "1 decimal place"
hit_list['Churn_Risk'] = hit_list['Churn_Risk'].map('{:.1%}'.format)

# Format Currency (e.g., 69.9 -> $69.90)
# .2f means "2 decimal floats", comma adds thousands separator
hit_list['MonthlyCharges'] = hit_list['MonthlyCharges'].map('${:,.2f}'.format)
hit_list['Expected_Loss'] = hit_list['Expected_Loss'].map('${:,.2f}'.format)

# 5. Show or Save
print(hit_list.head(10).to_string(index=False))
# hit_list.to_csv("churn_hit_list.csv", index=False) # Uncomment to save
#
#
#
#| label: logistic_regression_analysis
#| include: false

# ------------------------------------------------------------------
# 0. HANDLE MISSING VALUES (The Fix)
# ------------------------------------------------------------------
# Check if we have NaNs (likely in TotalCharges) and fill them with 0
# Logic: If tenure is 0, TotalCharges is blank -> implying 0 cost so far.
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Check to ensure clean
print("Missing values in X_train:", X_train.isnull().sum().sum())

# ------------------------------------------------------------------
# 1. SCALE THE DATA (Required for Logistic Regression)
# ------------------------------------------------------------------
# Logistic Regression gets confused if some numbers are huge (TotalCharges ~2000) 
# and others are tiny (SeniorCitizen 0 or 1). We scale them to be comparable.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 2. TRAIN MODEL
# ------------------------------------------------------------------
log_model = LogisticRegression(random_state=786)
log_model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------------
# 3. EXTRACT "ODDS RATIOS" (The Graph)
# ------------------------------------------------------------------
# Coefficients tell us direction. Exponentiating them gives "Odds Ratios".
# Odds Ratio > 1: Increases Churn Risk
# Odds Ratio < 1: Decreases Churn Risk
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_model.coef_[0]
})
coef_df['Odds_Ratio'] = np.exp(coef_df['Coefficient']) 
coef_df = coef_df.sort_values(by='Odds_Ratio', ascending=False)

# ------------------------------------------------------------------
# 4. PLOT THE ODDS RATIOS
# ------------------------------------------------------------------
plt.figure(figsize=(10, 8))

# Color code: Red for Bad (Increases Churn), Green for Good (Reduces Churn)
colors = ['#d62728' if x > 1 else '#2ca02c' for x in coef_df['Odds_Ratio']]

sns.barplot(x='Odds_Ratio', y='Feature', data=coef_df, palette=colors)

# Add a vertical line at 1 (The Neutral Line)
plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
plt.text(1.1, 0.5, 'Increases Risk -->', color='black')
plt.text(0.5, 0.5, '<-- Reduces Risk', color='black', ha='right')

plt.title('Logistic Regression: Odds Ratios\n(What multiplies the risk?)')
plt.xlabel('Odds Ratio (Multiplier)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 5. PRINT ACCURACY
# ------------------------------------------------------------------
acc = log_model.score(X_test_scaled, y_test)
print(f"Logistic Regression Accuracy: {acc:.2%}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: hit_list_table
#| echo: false

# 1. Filter for Active Customers (Churn == 0)
# We use the encoded 'Churn' column from your dataframe (0=No, 1=Yes)
active_cust = df[df['Churn'] == 0].copy()

# 2. Predict Probabilities
# We filter X based on the index of the active customers to ensure alignment
X_active = X.loc[active_cust.index] 
active_cust['Churn_Risk'] = rf_model.predict_proba(X_active)[:, 1]

# 3. Calculate Expected Loss
active_cust['Expected_Loss'] = active_cust['Churn_Risk'] * active_cust['MonthlyCharges']

# 4. Sort by Expected Loss to find the "Top 5"
top_hits = active_cust.sort_values(by='Expected_Loss', ascending=False).head(5)

# 5. Select and Rename Columns for the Report
display_table = top_hits[['customerID', 'Churn_Risk', 'MonthlyCharges', 'Expected_Loss']].copy()
display_table.columns = ['Customer ID', 'Churn Risk', 'Monthly Bill', 'Expected Loss']

# 6. Apply Formatting (Percentage and Currency)
display_table['Churn Risk'] = display_table['Churn Risk'].map('{:.1%}'.format)
display_table['Monthly Bill'] = display_table['Monthly Bill'].map('${:,.2f}'.format)
display_table['Expected Loss'] = display_table['Expected Loss'].map('${:,.2f} / mo'.format)

# 7. Print as Markdown
# Using display(Markdown(...)) ensures Quarto/Jupyter renders this as a formatted table
display(Markdown(display_table.to_markdown(index=False)))
display(Markdown(f"::: {{.keep_together}}\n{display_table}\n:::"))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
