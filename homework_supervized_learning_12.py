# Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

# Read the CSV file from the Resources folder into a Pandas DataFrame
file_path = Path("Resources/lending_data.csv")
df = pd.read_csv(file_path)

# Review the DataFrame
print(df)

# Separate the data into labels and features

# Separate the y variable, the labels
y = df['loan_status']

# Separate the X variable, the features
X = df.drop('loan_status', axis=1)

# Review the y variable Series
print(y)


0    0
1    0
2    0
3    0
4    0
Name: loan_status, dtype: int64

# Review the X variable DataFrame
print(X)


loan_size	interest_rate	borrower_income	debt_to_income	num_of_accounts	derogatory_marks	total_debt
0	10700.0	7.672	52800	0.431818	5	1	22800
1	8400.0	6.692	43600	0.311927	3	0	13600
2	9000.0	6.963	46100	0.349241	3	0	16100
3	10700.0	7.664	52700	0.430740	5	1	22700
4	10800.0	7.698	53000	0.433962	5	1	23000
Step 3: Check the balance of the labels variable (y) by using the value_counts function.

# Check the balance of our target values
print(y.value_counts())

0    75036
1     2500
Name: loan_status, dtype: int64
Step 4: Split the data into training and testing datasets by using train_test_split.

# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
logistic_model = LogisticRegression(random_state=1)

# Fit the model using training data
logistic_model = LogisticRegression(random_state=1)
logistic_model.fit(X_train, y_train)

LogisticRegression(random_state=1)
Step 2: Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.

# Make a prediction using the testing data
y_pred = logistic_model.predict(X_test)

# Print the balanced_accuracy score of the model
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(balanced_acc)

# Generate a confusion matrix for the model
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)

# Print the classification report for the model
class_report = classification_report(y_test, y_pred)
print(class_report)

                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384


# Import the RandomOverSampler module from imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

# Instantiate the random oversampler model
random_oversampler = RandomOverSampler(random_state=1)

# Fit the original training data to the random_oversampler model
X_train_resampled, y_train_resampled = random_oversampler.fit_resample(X_train, y_train)

# Count the distinct values of the resampled labels data
distinct_values = y_train_resampled.value_counts()


1    56271
0    56271
Name: loan_status, dtype: int64

# Instantiate the Logistic Regression model
logistic_model_resampled = LogisticRegression(random_state=1)

# Fit the model using the resampled training data
logistic_model_resampled.fit(X_train_resampled, y_train_resampled)

# Make a prediction using the testing data
y_pred_resampled = logistic_model_resampled.predict(X_test)

# Print the balanced_accuracy score of the model
balanced_acc_resampled = balanced_accuracy_score(y_test, y_pred_resampled)
print(balanced_acc_resampled)

# Generate a confusion matrix for the model
confusion_mat_resampled = confusion_matrix(y_test, y_pred_resampled)
print(confusion_mat_resampled)

array([[18649,   116],
       [    4,   615]])

# Print the classification report for the model
class_report_resampled = classification_report(y_test, y_pred_resampled)
print(class_report_resampled)
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384

