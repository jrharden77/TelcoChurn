Telco Customer Churn Analysis & Retention Strategy
Author: Joshua H.
Date: December 2025
Status: Phase 1 Complete (Predictive Modeling & Driver Analysis)

## Executive Overview

This project focuses on identifying "at-risk" customers within a telecommunications dataset to reduce churn and protect recurring revenue. By leveraging machine learning, we moved beyond simple descriptive statistics to building a predictive engine that flags potential churners with **83% accuracy (AUC)**.

The core output of this analysis is a **"Hit List" of 500 active, high-value customers** who are currently at risk of leaving, representing a potential revenue save of ~$30,000 - $40,000 per month.

## Key Insights & Drivers

Through Random Forest feature importance and Logistic Regression odds ratios, we identified three primary drivers of churn:

- **Price Sensitivity**: `MonthlyCharges` is the #1 predictor. Customers with bills >$80/month are disproportionately likely to cancel.

- **Contract Structure**: Customers on **Month-to-Month** contracts are the most volatile segment. Moving a user to a 1-year contract reduces churn risk significantly (Odds Ratio < 0.1).

- **Payment Friction**: Users paying via **Electronic Check** churn at nearly double the rate of those using credit card auto-pay.

##Methodology

### Data Preprocessing

- **Encoding**: Converted categorical variables (e.g., "Yes/No", Gender) into binary numeric formats (1/0).

- **Handling Missing Data**: Addressed missing values in `TotalCharges` and `OnlineBackup`.

- **Scaling**: Applied `StandardScaler` for distance-sensitive models like Logistic Regression.

### Models Implemented

- **Random Forest Classifier (Primary)**: Used for its high accuracy and ability to capture non-linear relationships.

  - **Strategy**: Implemented "Threshold Moving" (lowering the decision boundary to 30%) to prioritize **Recall**. This ensures we catch ~75% of churners, even at the cost of some false alarms.

- **Logistic Regression (Secondary)**:

  - Used to calculate **Odds Ratios**.

  - Provided the "multiplier effect" context (e.g., "Fiber Optic increases risk by 2.5x") for executive stakeholders.

## Project Structure

- `TelcoChurn.qmd`: The main Quarto analysis document. Generates the final PDF report.

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The raw dataset.

- `generate_hit_list.py`: Python script to generate the prioritized table of at-risk customers for the retention team.

- `executive_summary.md`: The text content for the final business report.

## Future Roadmap (Phase 2)

To further refine our retention strategy, the following techniques are planned for the next iteration:

- **K-Means Clustering (Customer Segmentation)**:

  - _Goal_: Move beyond "Churn vs. Stay" to understand who our customers are.

  - _Application_: Group customers into personas (e.g., "Young Techies," "Budget Seniors," "Families") to tailor marketing messages distinctively for each group.

- **K-Nearest Neighbors (KNN)**:

  - _Goal_: Recommendation and Similarity matching.

  - _Application_: "Look-alike" modeling. If Customer A churned, find the 5 customers most similar to Customer A who are still active and intervene immediately.

## Usage

This project is built using **Quarto** and **Python**. To generate the full report:

  1. Ensure required libraries are installed (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tabulate`).

  2. Run the Quarto render command:

> quarto render TelcoChurn.qmd --to pdf
