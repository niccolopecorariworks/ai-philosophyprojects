# Data Science and Ethics Project: K-Anonymity and Bias Mitigation - N. Pecorari, C. Monaco

## Overview
This project, developed for the 'Data Science and Ethics' course, addresses two critical aspects of ethical data science: privacy protection through K-Anonymity and bias mitigation in machine learning models.
The work utilizes the "Bank Marketing" dataset to demonstrate practical applications of these concepts.

## Introduction
The initial phase of the project focuses on safeguarding sensitive customer data from potential re-identification attacks. 
While the dataset lacks direct identifiers, combining quasi-identifiers could expose individual records. 
To counter this, K-Anonymity was implemented with a target k-value of 6, ensuring that each record is indistinguishable from at least five others.

## Key Steps:

- Quasi-Identifier Identification: age, job, and marital status were identified as quasi-identifiers.
- Generalization Strategies:
         - age: Generalized into broader ranges (19-40, 41-60, 61-87).
         - job: Grouped into three high-level categories (A: High-level, B: Manual/Technical, C: Others/Unemployed/Students).
         - marital: Suppressed entirely due to its limited informativeness and small number of unique values, which made generalization less effective.
- Verification: A k_anonimity control function was developed to verify the anonymization level at each step, confirming that k=6 was successfully achieved.

Outcome: The original quasi-identifier columns were removed, and generalized/suppressed versions were retained for subsequent machine learning tasks, ensuring data privacy while preserving utility.

## Bias Mitigation in Logistic Regression
The second part of the project tackles the issue of class imbalance in the target variable ('y', indicating term deposit subscription) to build a fair and effective Logistic Regression model. 
The dataset exhibited a significant imbalance (approximately 88% 'no' vs. 12% 'yes').

Methodology:

- Data Preparation: The dataset was split into training (50%), validation (25%), and test (25%) sets, ensuring stratification by the target variable. 
Categorical features were processed using One-Hot Encoding (drop='first'), and numerical features were scaled using StandardScaler.
- Initial Bias Mitigation: A Logistic Regression model was trained with class_weight='balanced'. While this improved recall for the minority class, precision remained low, indicating a high rate of false positives.
- Advanced Bias Mitigation (SMOTE): To address the precision issue, SMOTE (Synthetic Minority Oversampling Technique) was applied exclusively to the training set. 
A sampling strategy was chosen to balance the classes to an 80/20 ratio (majority/minority), preventing excessive oversampling and potential overfitting.

Outcome: The SMOTE-enhanced model significantly improved the precision of minority class predictions (over 53%) compared to the class_weight='balanced' approach. 
Although there was a slight trade-off in recall, the improved precision made SMOTE the preferred method for this project, aligning with the goal of reducing unproductive contacts in a marketing campaign.

## Tech stack
- Python
- pandas: Data manipulation and analysis
- matplotlib & seaborn: Data visualization
- scikit-learn: Machine learning model development (Logistic Regression, StandardScaler, OneHotEncoder, train_test_split)
- imblearn: Imbalanced-learn library for SMOTE
- pandasql: SQL-like queries on pandas DataFrames (for attack simulation)

## Notes
Developed ad a group assignment for the "Data Science and Ethics" course at Sapienza University of Rome.