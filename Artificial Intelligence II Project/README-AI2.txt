# Artificial Intelligence II Project - Android Malware Detection and Classification - N. Pecorari

## Overview
This project focuses on the detection and classification of Android malware (Malware vs. Goodware) using a Logistic Regression model. 
The dataset consists of various features extracted from Android applications, and the project addresses common challenges in machine learning, such as imbalanced datasets and feature engineering.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Results and Conclusion](#results-and-conclusion)
6. [How to Run](#how-to-run)

## 1. Introduction
Android malware poses a significant threat to mobile users. This project aims to build a robust classification model to accurately identify malicious applications, leveraging a comprehensive dataset of application features. 
I employ Logistic Regression, known for its interpretability and efficiency, especially when combined with careful data preparation.

## 2. Dataset
The dataset used in this project is `TUANDROMD.csv`, containing various static and dynamic features of Android applications. 
The target variable, `Label`, indicates whether an application is Malware (1) or Goodware (0).

## 3. Data Preprocessing and Feature Engineering
To prepare the data for modeling, the following steps were performed:

- **Handling Missing Values:** All `NaN` values were replaced with 0.
- **Zero Variance Feature Removal:** Features with zero variance (constant values) were identified and removed, as they provide no discriminative information for the model. This reduced the number of features significantly.
- **Multicollinearity Reduction:** A correlation analysis was performed to identify highly correlated features (absolute correlation > 0.8). An iterative approach was used to remove one feature from each highly correlated pair to mitigate multicollinearity, which can negatively impact model stability and interpretability.

After these steps, the dataset was reduced to **76 features**.

## 4. Model Training and Evaluation

### Data Splitting
The dataset was split into training, validation, and test sets to ensure robust model evaluation:
- **Training Set:** 70% of the data
- **Validation Set:** 15% of the data
- **Test Set:** 15% of the data
Stratified sampling was used to maintain the original class distribution in all splits.

### Cross-Validation with Logistic Regression
Two approaches were compared during cross-validation on the training set:

1.  **Logistic Regression with `class_weight='balanced'`:** The model was trained with `class_weight='balanced'` to address the class imbalance directly within the algorithm.
    - **Mean Accuracy:** ~0.9760
    - **Mean F1-score:** ~0.9848
    - **Mean AUC:** ~0.9973

2.  **Logistic Regression with SMOTE (Synthetic Minority Over-sampling Technique):** A pipeline was used where SMOTE was applied within each cross-validation fold to oversample the minority class (Goodware) before training the Logistic Regression model.
    - **Mean Accuracy:** ~0.9792
    - **Mean F1-score:** ~0.9869
    - **Mean AUC:** ~0.9973

The SMOTE approach showed slightly better overall performance, particularly in terms of F1-score, indicating a better balance between precision and recall.

### Final Model Training and Feature Importance
The final Logistic Regression model was trained on the entire training set, with SMOTE applied *prior* to training to achieve a balanced class distribution (50% Malware, 50% Goodware) in the training data.

**Top 10 Important Features (based on absolute Logistic Regression coefficients):**
1.  `VIBRATE`
2.  `RECEIVE_BOOT_COMPLETED`
3.  `RECEIVE_SMS`
4.  `SYSTEM_ALERT_WINDOW`
5.  `Ljava/net/URL;->openConnection`
6.  `SEND_SMS`
7.  `Landroid/hardware/Camera;->takePicture`
8.  `BATTERY_STATS`
9.  `BLUETOOTH`
10. `Ljava/lang/System;->load`

These features demonstrate the strongest linear relationship with the target variable, indicating their significance in distinguishing between malware and goodware.

### Model Evaluation on Test Set
Finally, the trained model was evaluated on the unseen test set:

- **Accuracy:** 0.9821
- **Classification Report:**
    - **Precision (Goodware/0):** 0.94
    - **Recall (Goodware/0):** 0.98
    - **F1-score (Goodware/0):** 0.96
    - **Precision (Malware/1):** 0.99
    - **Recall (Malware/1):** 0.98
    - **F1-score (Malware/1):** 0.99
- **AUC:** 0.9970

The high AUC score (0.9970) indicates excellent discriminative power of the model, successfully distinguishing between the two classes. The confusion matrix further validates the model's performance with a low number of misclassifications.

## 5. Results and Conclusion
The Logistic Regression model, particularly when combined with SMOTE for handling class imbalance and robust feature engineering, demonstrates excellent performance in classifying Android applications as either malware or goodware. The identified important features provide valuable insights into the characteristics that differentiate malicious software.

This project provides a solid foundation for developing effective Android malware detection systems, highlighting the importance of data preprocessing and appropriate techniques for imbalanced datasets.

## 6. How to Run
1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Ensure Google Drive Access:** The notebook expects the `TUANDROMD.csv` file to be located in your Google Drive at `/content/drive/My Drive/TUANDROMD.csv`. Make sure to mount your Google Drive in Colab and upload the dataset there.
3.  **Install Dependencies:** All necessary libraries are standard in Google Colab (pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn).
4.  **Run the Jupyter Notebook:** Open and run the `your_notebook_name.ipynb` notebook cell by cell.
