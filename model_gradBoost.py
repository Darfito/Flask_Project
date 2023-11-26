from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt 
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, classification_report)
from sklearn.feature_selection import f_classif
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Load the data
# response = pd.read_csv("file:///Users/naufalazis/Documents/KLYH/0%20COOLYEAH/SEMESTER_5/AVD%20PRAK/UTS/responses.csv")
response = pd.read_csv("responses.csv")

# Save column names before encoding
column_names = response.columns.tolist()

# Create an encoder instance
encoder = OrdinalEncoder()

# Select categorical columns (2nd column, 5th to 11th column)
categorical_columns = response.columns[1:2].tolist() + response.columns[4:11].tolist()

# Encode the selected categorical columns
response[categorical_columns] = encoder.fit_transform(response[categorical_columns])

# encoding description
print()
print("==============Encoding Description===============")
encoded_categories = encoder.categories_
for col, categories in zip(categorical_columns, encoded_categories):
    print(f'Keterangan encoding untuk kolom {col}:')
    for i, category in enumerate(categories):
        print(f'{i} = {category}')
print()

# Concatenate the encoded categorical columns with the rest of the dataset
# Assuming the rest of the columns are non-categorical
print("==============Concatenated Value===============")
response = pd.concat([response[categorical_columns], response.drop(categorical_columns, axis=1)], axis=1)

# Print the modified DataFrame
print(response)

from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Group the variables
X = response.iloc[:, [3,4,5]]
y = response.iloc[:, 7]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the data using z-score
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=42)

# Train the model
gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 5))
plt.title('Confusion Matrix')
sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Save the Gradient Boosting model
joblib.dump(gb_classifier, 'gb_model.joblib')

# Save the scaler for later use
joblib.dump(scaler, 'scaler.joblib')

import pickle
# Simpan model ke file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(gb_classifier, model_file)
