**Importing the libraries necessary for loading and reading the dataset**

We import "drive" which allows us to retrieve the desired file from Google Drive's cloud. For exploratory data analysis, we will use "pandas", "numpy", "matplotlib" and "seaborn" which will allow us to operate directly on the dataset and generate graphs. We import "pandasql" for what will be a data retrieval simulation and "re" useful for string manipulation and regular expressions.
"""

# Import necessary libraries
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install pandasql
import pandasql as psql
import re

"""**Loading the dataset and first data exploration**

Let's proceed with loading the dataset and analyze its structure (number of rows, number of columns, number of missing values for each feature, list of constant features).
"""

# DF Loading

file_path = '/content/drive/My Drive/bank.csv'
df = pd.read_csv(file_path, sep=';')
print(df.head().to_string())

# Feature and target definition
target = 'y'
feat_cols = [col for col in df.columns if col != target]

# DF Structure
n_rows, n_cols = df.shape
print(f"Number of rows: {n_rows}")
print(f"Number of columns: {n_cols}")

# Number of NaNs
nan_count = df.isnull().sum()
print("Number of NaNs per feature:")
print(nan_count)

# Number of constant features
constant_features = df.columns[df.nunique() == 1]
print("Number of constant features:", len(constant_features))
if len(constant_features>=1):
  constant_list = [col for col in df.columns if df[col].nunique() == 1]
  print("List of constant features:", constant_list)

"""**Description of the "Bank Marketing" dataset**

The "Bank Marketing" dataset refers to a marketing campaign (via phone call) by a Portuguese bank. The version used is available at the following link: https://archive.ics.uci.edu/dataset/222/bank+marketing.
Below is the description of the features:
1) age: age of the customer
2) job: type of job of the customer
3) marital: marital status of the customer
4) education: educational background of the customer
5) default: indicates if the individual has a credit in default (unpaid debt)
6) balance: average annual balance on the account (in euros)
7) housing: whether the customer has a "housing loan"
8) loan: whether the customer has an active personal loan
9) contact: contact method for the call
10) day: day of the last contact
11) month: month of the last contact
12) duration: duration of the last contact in seconds
13) campaign: number of contacts made
14) pdays: Number of days passed since the contact with that client in a previous campaign (value -1 indicates the client has not been contacted before)
15) previous: Number of contacts made before the current campaign for that client
16) poutcome: Outcome of the previous marketing campaign
17) y: target. Whether the client subscribed to a term deposit ("yes") or not ("no")

**Privacy Protection in case of an attack based on background knowledge**

Suppose an individual with malicious intent operates to extract a series of information from our database. Fortunately, we do not possess direct identifiers such as the customer's name, surname, or address, but our information, if combined with others available in other archives, can lead to the re-identification of the registered customer. We provide as an example the case in which the cybercriminal wants to extract all records relating to 30-year-old married individuals working in management.
"""

# Simulation of a data mining attack based on background knowledge

query = "SELECT * FROM df WHERE age=30 AND job='management' AND marital='married'"

query_result = psql.sqldf(query, globals())
display(query_result)

"""**Attack Mitigation: K-Anonymity**

To counter the attack, it is necessary to provide an adequate level of protection and, at the same time, try not to lose a high amount of information (which will be important for the development of a Machine Learning model). Considering the sensitivity of the information we possess, achieving a level of K-Anonymity with **k=6** can be a good compromise. Such a level of K-Anonymity implies that each individual in the dataset is indistinguishable from at least k-1 other individuals based on the so-called quasi-identifiers. In other words, **each individual is "hidden" within a group composed of at least 6 people with the same identifiable characteristics**, thus making it impossible for the cybercriminal to determine with certainty which of the 6 people the extracted data belongs to.

Let's proceed with the selection of quasi-identifiers and create a K-Anonymity control function.
"""

# Selection of quasi-identifiers
q_id = ['age', 'job', 'marital']
df_an = df.copy()

# K-ANONYMITY control function
def k_anonimity(df_c,k_val,q_id_feat):
  if df_c.empty:
    return False
  exst_q_id_feat = [col for col in q_id_feat if col in df_c.columns]
  if not exst_q_id_feat:
    return False
  counts = df_c.groupby(exst_q_id_feat, observed=False).size()
  if counts.empty:
    return False
  return (counts>=k_val).all()

q_id_gen = ['age_gen', 'job_gen', 'marital_gen']
df_an['age_gen'] = df['age'].astype(str)
df_an['job_gen'] = df['job']
df_an['marital_gen'] = df['marital']

display(df_an.head())

"""**First Generalization: Age ('age')**

To generalize the first quasi-identifier, we calculate the minimum and maximum values contained in the age column. Based on the results, we will elaborate age ranges into which each value will be mapped.
"""

# Age generalization
age_min = df_an['age'].min()
age_max = df_an['age'].max()

print(f"Minimum age: {age_min}")
print(f"Maximum age: {age_max}")

"""Let's proceed with the creation of a function that maps "age" values into three ranges:
1) 19-40
2) 41-60
3) 61-87
"""

# Age generalization function with specific ranges
def age_generalization(year):
    if 19 <= year <= 40:
        return '19-40'
    elif 41 <= year <= 60:
        return '41-60'
    elif 61 <= year <= 87:
        return '61-87'
    else:
        return 'Other' # Handle any ages outside the defined ranges

"""Before activating the function, let's check with k=3 if the dataset is already anonymized."""

# Initial check with k = 3
k = 3
check_1 = k_anonimity(df_an, k, q_id_gen)
if check_1:
  print(f'The dataset is already anonymized with k = {k}')
  get_an = True
else:
  print(f'The dataset is not anonymized with k = {k}')
  get_an = False

"""Given the result, let's proceed with the generalization of age and check if we have reached a minimum level of anonymization."""

# Age generalization
df_an['age_gen'] = df['age'].apply(lambda x: age_generalization(x))

# K-anonymity check after age generalization
q_id_gen_new = ['age_gen', 'job_gen', 'marital_gen']
check_1 = k_anonimity(df_an, k, q_id_gen_new)
if check_1:
  print(f'The dataset is anonymized with k = {k} after age generalization.')
  get_an = True
else:
  print(f'The dataset is not anonymized with k = {k} after age generalization.')
  get_an = False

"""Having not obtained the desired result, we continue by treating the second of the chosen quasi-identifiers: "job".

**Second Generalization: Job ("job")**

To understand how we can generalize the values contained in "job" without losing too much informativeness, we print the unique values contained in the column.
"""

unique_jobs = df_an['job_gen'].unique().tolist()
print(unique_jobs)

num_unique_jobs = df_an['job_gen'].nunique()
print(f"\nNumber of unique values in the 'job' column: {num_unique_jobs}")

"""The results show a list composed of twelve job types. We decide to create a function that maps the various types into three categories:
1) A: High-level, decision-making, managerial jobs.
2) B: Manual, technical, industrial jobs.
3) C: Students, unemployed, retired, and remaining types.

Let's apply the function and project the result to see if it was successful.
"""

def generalize_job(job):
    job_mapping = {
        'management': 'A',
        'admin.': 'A',
        'entrepreneur': 'A',
        'self-employed': 'A',
        'blue-collar': 'B',
        'services': 'B',
        'technician': 'B',
        'housemaid': 'C',
        'retired': 'C',
        'unemployed': 'C',
        'student': 'C',
        'unknown': 'C'
    }
    return job_mapping.get(job, job)

# Apply the function
df_an['job_generalized'] = df_an['job_gen'].apply(generalize_job)

# Display the results
display(df_an[['job_gen', 'job_generalized']].head())

"""Let's update the list of quasi-identifiers with the new generalizations and proceed with the k-anonymity check."""

# K-anonymity check
q_id_gen_new2 = ['age_gen', 'job_generalized', 'marital_gen']
check_1 = k_anonimity(df_an, k, q_id_gen_new2)
if check_1:
  print(f'The dataset is already anonymized with k = {k}')
else:
  print(f'The dataset is not anonymized with k = {k}')

"""The result shows that further generalization may be necessary. Let's proceed, therefore, by treating the "marital" feature.

**Third Generalization: Marital Status ("marital")**

As with "job", let's print all unique values contained in the marital status column.
"""

unique_marital = df['marital'].unique().tolist()
print(unique_marital)

num_unique_marital = df['marital'].nunique()
print(f"\nNumber of unique values in the 'marital' column: {num_unique_marital}")

"""The results show only three values. Considering domain knowledge, we can proceed with the suppression of the information contained in the column and print the result to verify that the operation was successful."""

# Suppression of the 'marital_gen' column by replacing values with 'Suppressed'
df_an['marital_gen'] = 'Suppressed'

# Display the first rows to verify suppression
display(df_an[['marital_gen']].head())

"""Having verified the operation, we proceed with the K-Anonymity check."""

# K-anonymity check
q_id_gen_new3 = ['age_gen', 'job_generalized', 'marital_gen']
check_1 = k_anonimity(df_an, k, q_id_gen_new3)
if check_1:
  print(f'The dataset is already anonymized with k = {k}')
else:
  print(f'The dataset is not anonymized with k = {k}')

# K-anonymity check with k=6
k_test = 6
q_id_gen_for_test = ['age_gen', 'job_generalized', 'marital_gen']

check_k4 = k_anonimity(df_an, k_test, q_id_gen_for_test)

if check_k4:
  print(f'The dataset is anonymized with k = {k_test} after generalization and suppression.')
else:
  print(f'The dataset is not anonymized with k = {k_test} after generalization and suppression.')

k_test_ = 7
q_id_gen_for_test = ['age_gen', 'job_generalized', 'marital_gen']

check_k4 = k_anonimity(df_an, k_test_, q_id_gen_for_test)

if check_k4:
  print(f'The dataset is anonymized with k = {k_test_} after generalization and suppression.')
else:
  print(f'The dataset is not anonymized with k = {k_test_} after generalization and suppression.')

"""As the results show, we have achieved the desired level of K-Anonymity. In this way, as we said a few paragraphs back, each individual is indistinguishable from 5 others within the group based on the quasi-identifiers. To further verify the success of the k-anonymization operation, we can observe the groupings in the results of the next cell."""

# Manual K-anonymity verification

q_id_final = ['age_gen', 'job_generalized', 'marital_gen']

group_sizes = df_an.groupby(q_id_final, observed=False).size()

print("Equivalence group sizes:")
print(group_sizes.sort_values().head())

k_value_check = 6
min_group_size = group_sizes.min()

print(f"\nMinimum group size: {min_group_size}")

if min_group_size >= k_value_check:
    print(f"Manual verification successful: Minimum group size is >= {k_value_check}")
else:
    print(f"Manual verification failed: Minimum group size is < {k_value_check}")

"""Goal achieved. We can proceed with creating a copy of the dataset necessary for the development of a Machine Learning model and print the result to verify that the operation was successful."""

# Create a clean copy of df_an for machine learning
df_ml = df_an.copy()

# Define original non-generalized columns to remove
original_quasi_identifiers = ['age', 'job', 'job_gen', 'marital']

# Remove original non-generalized columns from df_ml
df_ml = df_ml.drop(columns=original_quasi_identifiers)

# Display the first rows of the new DataFrame to verify cleaning
print("DataFrame for Machine Learning (df_ml) - First rows:")
display(df_ml.head())

# Display the columns of the new DataFrame to confirm removal
print("\nColumns of df_ml DataFrame:")
print(df_ml.columns.tolist())

"""# **Logistic Regression and BIAS Mitigation**

**EDA**

Let's start the new section by defining features and target through two variables. Subsequently, we visualize the distribution of the target classes. Based on the result, we will decide whether to implement balancing techniques or keep the proportions unchanged.
"""

# Feature and target definition
target_y = 'y'
feat_cols = [col for col in df_ml.columns if col != target_y]

# Visualization of the distribution of the target column 'y'
plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=df_ml, palette='viridis', hue='y', legend=False) # Use hue to color bars
plt.title('Distribution of the Target Variable (y)')
plt.xlabel('Customer Response')
plt.ylabel('Count')
plt.show()

# Calculation of the percentage of each class in the target column 'y'
target_percentage = df_ml['y'].value_counts(normalize=True) * 100

print("Percentage of classes in the target column 'y':")
print(target_percentage)

"""The results show a huge imbalance between the classes. The majority class ('no'), considering the goal of developing a logistic regression model, will likely be more favored in predictions at the expense of the minority class. This **bias** undoubtedly compromises the model's generalization effectiveness, so it will be necessary to implement one or more mitigation techniques.

Before proceeding with bias treatment, let's examine the importance of numerical and categorical features in relation to the 'y' target. This operation is useful to check if it will be necessary to exclude some features due to lack of impact on predictions or low informativeness. Keeping uninformative features could slow down or "confuse" the model in processing training data and making predictions on the test set.
"""

# Selection of relevant numerical columns for EDA
numerical_features = ['balance', 'duration', 'campaign', 'pdays', 'previous']

# Histograms for numerical features based on target 'y'
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_ml, x=feature, hue='y', multiple='stack', palette='viridis', kde=True)
    plt.title(f'Distribution (Histogram) of {feature} for the Target Variable (y)')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# Selection of relevant categorical columns for EDA
categorical_features = ['education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'age_gen', 'job_generalized', 'marital_gen']

# Countplots for categorical features
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, data=df_ml, palette='viridis', hue=feature, legend=False)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Selection of relevant categorical columns for EDA
categorical_features = ['education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'age_gen', 'job_generalized', 'marital_gen']

print("Proportion of 'yes' for each category:")

for feature in categorical_features:
    # Calculate the proportion of 'yes' for each category
    # Convert 'y' to numeric (1 for 'yes', 0 for 'no') to calculate the mean
    df_ml['y_numeric'] = df_ml['y'].apply(lambda x: 1 if x == 'yes' else 0)
    proportion_yes = df_ml.groupby(feature)['y_numeric'].mean().sort_values(ascending=False) * 100

    print(f"\n--- {feature} ---")
    print(proportion_yes)

# Removal of the temporary y_numeric column
df_ml = df_ml.drop(columns=['y_numeric'])

"""The results for the numerical features show how the features "duration" (call duration), "pdays" (number of days since the last client contact), and "previous" (number of contacts made before the current campaign) are highly predictive. As for the categorical features, there don't seem to be significant correlations with the positive target, with the exception of the "month" feature and the value "october," which appears to be the month with the highest percentage of "yes," and "poutcome," a feature indicating whether the previous campaign was successful, which inevitably leads to a high percentage of positive outcomes if the previous campaign was successful. All other categorical features do not show high correlation but can still be informative.

Based on the anonymization results from the previous section and these latest analyses, we can consider the "marital_gen" feature (the client's marital status) to have very little informativeness. Therefore, let's proceed with its removal.
"""

# Removal of the marital_gen column
df_ml = df_ml.drop(columns=['marital_gen'])

print(df_ml.head())

"""**Splitting into train, validation, and test sets**

Before implementing **bias** mitigation strategies, it is necessary to immediately split the dataset into train, validation, and test sets. This is to avoid data leakage between the various partitions that could compromise training and prediction quality.

We proceed by first splitting a training set and a temporary set 50/50. Then we split the temporary set into two equal parts to have 25% of the original total for both.
"""

# Separation of features (X) and target (y)
X = df_ml.drop('y', axis=1)
y = df_ml['y']

print("Shape of X (features):", X.shape)
print("Shape of y (target):", y.shape)

from sklearn.model_selection import train_test_split

# First split: Training and a temporary set (Test + Validation)
# We split so that the temporary set is 50% to get Test and Val at 25% each
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

# Second split: Splitting the temporary set into Validation and Test
# We split the temporary set 50/50 (0.5) to get Test and Validation at 25% each of the original total
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print set shapes for verification
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of y_test:", y_test.shape)

"""**Correlation matrix of numerical features (train set)**

Let's elaborate a correlation matrix that allows us to check if there are highly correlated features. From the results, we can decide whether to eliminate one of the features within a highly correlated pair to ensure a cleaner dataset for the model.
"""

# Selection of numerical columns from the training set
numerical_cols = X_train.select_dtypes(include=np.number).columns

# Calculation of the correlation matrix for numerical features
correlation_matrix = X_train[numerical_cols].corr()

# Correlation matrix (heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features (X_train)')
plt.show()

"""With a correlation score of 0.60, the features `pdays` and `previous` are the most correlated pair. In this case, we are not in the presence of an excessively significant correlation. Therefore, without risking loss of informativeness, we can keep both features.

**Encoding categorical features (One-Hot Encoding)**

Let's proceed with One-Hot Encoding of categorical features. This will generate new boolean features corresponding to the values of the categorical features. We use the "drop='First'" option to avoid multicollinearity. We "fit" the encoding on the train set and use "transform" on the validation and test sets. Once we have the names of the new encoded columns, we convert the arrays to dataframes and restore the original indices.
"""

# Identification of categorical and numerical columns
categorical_features = X_train.select_dtypes(include='object').columns
numerical_features = X_train.select_dtypes(include=np.number).columns

print(f"Identified categorical columns: {list(categorical_features)}")
print(f"Identified numerical columns: {list(numerical_features)}")

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Initialization of OneHotEncoder
# handle_unknown='ignore' is useful if there are categories in val/test sets not seen in train
# drop='first' to avoid perfect collinearity
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

# Fit the encoder exclusively on the training set
one_hot_encoder.fit(X_train[categorical_features])

# Transformation of training, validation, and test sets
X_train_cat_encoded = one_hot_encoder.transform(X_train[categorical_features])
X_val_cat_encoded = one_hot_encoder.transform(X_val[categorical_features])
X_test_cat_encoded = one_hot_encoder.transform(X_test[categorical_features])

# Get the names of the new encoded columns
new_categorical_features = list(one_hot_encoder.get_feature_names_out(categorical_features))

# Convert encoded arrays to DataFrames and restore original indices
X_train_cat_encoded_df = pd.DataFrame(X_train_cat_encoded, columns=new_categorical_features, index=X_train.index)
X_val_cat_encoded_df = pd.DataFrame(X_val_cat_encoded, columns=new_categorical_features, index=X_val.index)
X_test_cat_encoded_df = pd.DataFrame(X_test_cat_encoded, columns=new_categorical_features, index=X_test.index)

print("Shape of X_train_cat_encoded_df:", X_train_cat_encoded_df.shape)
print("Shape of X_val_cat_encoded_df:", X_val_cat_encoded_df.shape)
print("Shape of X_test_cat_encoded_df:", X_test_cat_encoded_df.shape)

print("\nFirst rows of X_train_cat_encoded_df:")
display(X_train_cat_encoded_df.head())

"""**Scaling numerical features (Standard Scaler)**

For numerical features, we use a Standard Scaler which allows us to scale the features to have an approximately zero mean and a standard deviation close to 1. We use the scaler because logistic regression, like other algorithms, is sensitive to feature scaling. In other words, features with larger values than others, if not scaled, can dominate and disproportionately influence model decisions, and the algorithm might delay convergence.
"""

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler exclusively on the training set (numerical columns)
scaler.fit(X_train[numerical_features])

# Transform the training, validation, and test sets
X_train_scaled = scaler.transform(X_train[numerical_features])
X_val_scaled = scaler.transform(X_val[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Convert scaled arrays to DataFrames and restore original indices
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=numerical_features, index=X_val.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)

print("Shape of X_train_scaled_df:", X_train_scaled_df.shape)
print("Shape of X_val_scaled_df:", X_val_scaled_df.shape)
print("Shape of X_test_scaled_df:", X_test_scaled_df.shape)

print("\nFirst rows of X_train_scaled_df:")
display(X_train_scaled_df.head())

"""Let's now proceed with concatenating the encoded categorical features and the scaled numerical features, and print the result to see if the operation was successful."""

# Concatenation of encoded categorical features and scaled numerical features
X_train_final = pd.concat([X_train_cat_encoded_df, X_train_scaled_df], axis=1)
X_val_final = pd.concat([X_val_cat_encoded_df, X_val_scaled_df], axis=1)
X_test_final = pd.concat([X_test_cat_encoded_df, X_test_scaled_df], axis=1)

print("Final shape of X_train_final:", X_train_final.shape)
print("Final shape of X_val_final:", X_val_final.shape)
print("Final shape of X_test_final:", X_test_final.shape)

print("\nFirst rows of X_train_final:")
display(X_train_final.head())

"""**Logistic Regression Training with class_weight='balanced'**

To mitigate the bias caused by class imbalance, before using oversampling or undersampling techniques, we can try to start training with the option **class_weight='balanced'**. This parameter analyzes the class distribution and assigns higher weights to the minority class and lower weights to the majority class based on this. In doing so, during training, the model calculates the cost function, and errors made on minority class samples will be given more weight, encouraging the model to pay more attention to less numerous samples.
"""

from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression model with class_weight='balanced'
logistic_model_balanced = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced')

# Train the model on the pre-processed original training set (no resampling)
print("Training a new Logistic Regression model with class_weight='balanced'...")
logistic_model_balanced.fit(X_train_final, y_train)
print("Training completed.")

"""Once training is complete, we can proceed to print the metrics."""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Evaluation on the Training set for logistic_model_balanced
print("--- Evaluation on the Training Set (class_weight='balanced') ---")

# Predictions on the training set using the balanced model
y_train_pred_balanced = logistic_model_balanced.predict(X_train_final)
y_train_prob_balanced = logistic_model_balanced.predict_proba(X_train_final)[:, 1] # Probability of the positive class ('yes')

# Calculate metrics
accuracy_train_balanced = accuracy_score(y_train, y_train_pred_balanced)
precision_train_balanced = precision_score(y_train, y_train_pred_balanced, pos_label='yes')
recall_train_balanced = recall_score(y_train, y_train_pred_balanced, pos_label='yes')
f1_train_balanced = f1_score(y_train, y_train_pred_balanced, pos_label='yes')
roc_auc_train_balanced = roc_auc_score(y_train, y_train_prob_balanced)

# Print metrics
print(f"Accuracy: {accuracy_train_balanced:.4f}")
print(f"Precision: {precision_train_balanced:.4f}")
print(f"Recall: {recall_train_balanced:.4f}")
print(f"F1-Score: {f1_train_balanced:.4f}")
print(f"AUC-ROC: {roc_auc_train_balanced:.4f}")

# Classification report for a detailed summary
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred_balanced))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_train_pred_balanced))


print("\n" + "="*50 + "\n") # Separator

# Evaluation on the Validation set for logistic_model_balanced
print("--- Evaluation on the Validation Set (class_weight='balanced') ---")

# Predictions on the validation set
y_val_pred_balanced = logistic_model_balanced.predict(X_val_final)
y_val_prob_balanced = logistic_model_balanced.predict_proba(X_val_final)[:, 1] # Probability of the positive class ('yes')

# Calculate metrics
accuracy_val_balanced = accuracy_score(y_val, y_val_pred_balanced)
precision_val_balanced = precision_score(y_val, y_val_pred_balanced, pos_label='yes')
recall_val_balanced = recall_score(y_val, y_val_pred_balanced, pos_label='yes')
f1_val_balanced = f1_score(y_val, y_val_pred_balanced, pos_label='yes')
roc_auc_val_balanced = roc_auc_score(y_val, y_val_prob_balanced)

# Print metrics
print(f"Accuracy: {accuracy_val_balanced:.4f}")
print(f"Precision: {precision_val_balanced:.4f}")
print(f"Recall: {recall_val_balanced:.4f}")
print(f"F1-Score: {f1_val_balanced:.4f}")
print(f"AUC-ROC: {roc_auc_val_balanced:.4f}")

# Classification report for a detailed summary
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_balanced))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_balanced))


print("\n" + "="*50 + "\n") # Separator


# Evaluation on the Test set for logistic_model_balanced
print("--- Evaluation on the Test Set (class_weight='balanced') ---")

# Predictions on the test set
y_test_pred_balanced = logistic_model_balanced.predict(X_test_final)
y_test_prob_balanced = logistic_model_balanced.predict_proba(X_test_final)[:, 1] # Probability of the positive class ('yes')

# Calculate metrics
accuracy_test_balanced = accuracy_score(y_test, y_test_pred_balanced)
precision_test_balanced = precision_score(y_test, y_test_pred_balanced, pos_label='yes')
recall_test_balanced = recall_score(y_test, y_test_pred_balanced, pos_label='yes')
f1_test_balanced = f1_score(y_test, y_test_pred_balanced, pos_label='yes')
roc_auc_test_balanced = roc_auc_score(y_test, y_test_prob_balanced)

# Print metrics
print(f"Accuracy: {accuracy_test_balanced:.4f}")
print(f"Precision: {precision_test_balanced:.4f}")
print(f"Recall: {recall_test_balanced:.4f}")
print(f"F1-Score: {f1_test_balanced:.4f}")
print(f"AUC-ROC: {roc_auc_test_balanced:.4f}")

# Classification report for a detailed summary
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_balanced))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_balanced))

"""The metrics do not show signs of effectiveness in bias mitigation. Although the Accuracy level on the test set is very high (0.8161), the other metrics do not alleviate the initial concern.

Looking at the confusion matrix of the test set, 825 are correctly predicted true negatives and 98 are true positives. The most relevant data is the number of false positives, 175, which exceeds the number of true negatives. The model, therefore, has high precision (0.96) in predicting the majority class of "no" and very low precision (0.36) for the minority class of "yes".

Given the results, we can try to implement a more "incisive" technique to balance the classes and start training again. Considering the current state and the nature of the data, it becomes a priority to try to increase the precision and the remaining metrics of the minority class (possibly over 50%).

**Class balancing with SMOTE**

We import the necessary library to correctly execute **SMOTE**, an oversampling technique that creates "synthetic" samples for the minority class. Contrary to a RandomOversampler, which simply duplicates instances, SMOTE generates new samples after selecting the nearest neighbors for each sample in the minority class. Neighboring samples are connected by a "line" and SMOTE moves along this line to create a new synthetic sample, repeating the operation until the desired balance is achieved.

To avoid data leakage, we apply SMOTE **only to the training set**. We have chosen a 80-20 proportion as a baseline (compared to the almost 90-10 of the original distribution). In this way, we do not have excessive oversampling that could lead to overfitting. If we do not achieve the desired results, we can consider increasing the proportion.
"""

!pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

# Initialize SMOTE

# Calculate the desired number of samples for the minority class ('yes')
# so that the ratio is 80% 'no' and 20% 'yes'.
# In our case, 'no' is the majority class.
class_counts = y_train.value_counts()
n_majority = class_counts['no']
n_minority_desired = int(n_majority * (20 / 80)) # Calculate the number of samples for the minority class

# Define the sampling strategy as a dictionary
sampling_strategy = {'no': n_majority, 'yes': n_minority_desired}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)


# Apply SMOTE exclusively to the training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)

# Print the new shapes and class distribution in the balanced training set
print("Shape of X_train_resampled after SMOTE (80/20):", X_train_resampled.shape)
print("Class distribution in y_train_resampled after SMOTE (80/20):")
print(y_train_resampled.value_counts())

"""**Logistic Regression Training with SMOTE**

After applying SMOTE to the train set, we can proceed with the new training and print the metrics.
"""

from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model for the resampled training set
logistic_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000) # Use solver='liblinear' for datasets of this size and increased max_iter

# Train the model on the resampled training set
print("Training the Logistic Regression model...")
logistic_model.fit(X_train_resampled, y_train_resampled)
print("Training completed.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Evaluation on the Training set (SMOTE 80/20)
print("--- Evaluation on the Training Set (SMOTE 80/20) ---")

# Predictions on the training set (SMOTE 80/20)
y_train_resampled_pred = logistic_model.predict(X_train_resampled)
y_train_resampled_prob = logistic_model.predict_proba(X_train_resampled)[:, 1] # Probability of the positive class ('yes')

# Calculate metrics on the training set (SMOTE 80/20)
accuracy_train_resampled = accuracy_score(y_train_resampled, y_train_resampled_pred)
precision_train_resampled = precision_score(y_train_resampled, y_train_resampled_pred, pos_label='yes')
recall_train_resampled = recall_score(y_train_resampled, y_train_resampled_pred, pos_label='yes')
f1_train_resampled = f1_score(y_train_resampled, y_train_resampled_pred, pos_label='yes')
roc_auc_train_resampled = roc_auc_score(y_train_resampled, y_train_resampled_prob)

# Print metrics
print(f"Accuracy: {accuracy_train_resampled:.4f}")
print(f"Precision: {precision_train_resampled:.4f}")
print(f"Recall: {recall_train_resampled:.4f}")
print(f"F1-Score: {f1_train_resampled:.4f}")
print(f"AUC-ROC: {roc_auc_train_resampled:.4f}")

# Classification report for a detailed summary
print("\nClassification Report:")
print(classification_report(y_train_resampled, y_train_resampled_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_train_resampled, y_train_resampled_pred))

print("\n" + "="*50 + "\n") # Separator


# Evaluation on the Validation set
print("--- Evaluation on the Validation Set (SMOTE 80/20) ---")

# Predictions on the validation set
y_val_pred = logistic_model.predict(X_val_final)
y_val_prob = logistic_model.predict_proba(X_val_final)[:, 1] # Probability of the positive class ('yes')

# Calculate metrics
accuracy_val = accuracy_score(y_val, y_val_pred)
precision_val = precision_score(y_val, y_val_pred, pos_label='yes')
recall_val = recall_score(y_val, y_val_pred, pos_label='yes')
f1_val = f1_score(y_val, y_val_pred, pos_label='yes')
roc_auc_val = roc_auc_score(y_val, y_val_prob)

# Print metrics
print(f"Accuracy: {accuracy_val:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"F1-Score: {f1_val:.4f}")
print(f"AUC-ROC: {roc_auc_val:.4f}")

# Classification report for a detailed summary
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


print("\n" + "="*50 + "\n") # Separator


# Evaluation on the Test set
print("--- Evaluation on the Test Set (SMOTE 80/20) ---")

# Predictions on the test set
y_test_pred = logistic_model.predict(X_test_final)
y_test_prob = logistic_model.predict_proba(X_test_final)[:, 1] # Probability of the positive class ('yes')

# Calculate metrics
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred, pos_label='yes')
recall_test = recall_score(y_test, y_test_pred, pos_label='yes')
f1_test = f1_score(y_test, y_test_pred, pos_label='yes')
roc_auc_test = roc_auc_score(y_test, y_test_prob)

# Print metrics
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")
print(f"AUC-ROC: {roc_auc_test:.4f}")

# Classification report for a detailed summary
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

"""We have obtained the desired results. Thanks to the SMOTE technique, despite the still wide gap between the precision of the negative class prediction and the positive class, we have increased the precision for the minority class to over 53%.

From the confusion matrix, we note that the model correctly predicted the positive class 66 times and incorrectly classified the positive class as negative 59 times. The metrics for the majority class remain, not surprisingly, very high.

**"Balanced" vs SMOTE: Final Considerations**

The model with "class_weight='balanced'" showed a very high Recall score on the test set (0.75). This means it was very good at identifying most of the customers who subscribed to a deposit with the bank. However, the Precision for the minority class is very low (0.36), classifying many customers as depositors even if they belong to the opposite class.

The SMOTE model, on the other hand, achieved higher Precision in the minority class compared to the previous model. With a score of 0.53, it shows that it reduced the number of false positives. The Recall, compared to the class_weight model, is lower (0.50), as it identified a smaller percentage of true positives.

The choice between the two models therefore depends on the specific objective of the marketing campaign.
If the goal is to identify the largest number of potential depositors (even at the cost of contacting some who are not), the model with `class_weight='balanced'` could be favored due to its Recall.
If, instead, it is more important to reduce the number of unproductive contacts (reduce false positives), the SMOTE model is better due to its high Precision.

Considering, however, our initial bias mitigation objective, choosing the model with the most stable and balanced metrics appears to be a sensible choice. Therefore, SMOTE, thanks to the improvement of the metrics, is configured as the first choice.
"""
