from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

from sklearn.model_selection import cross_val_score
import pandas as pd

# Load the dataset
file_path = 'spamTrain1.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the data and information about its structure
data.head(), data.info()

# Separate features and target variable
X = data.iloc[:, :-1]  # all columns except the last one
y = data.iloc[:, -1]   # the last column

# Check class balance
class_counts = y.value_counts(normalize=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions and calculate metrics
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])

# Cross-validation accuracy
cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')

# Display the results
class_counts, accuracy, precision, recall, roc_auc, cv_scores.mean()
