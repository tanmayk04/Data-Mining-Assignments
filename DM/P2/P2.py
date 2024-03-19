import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Step 1: Load the dataset
data = pd.read_csv("NBAstats.csv")

# Step 2: Preprocess the data

# Convert categorical variable 'Pos' to numerical representation
positions = {
    'SG': 0,
    'PG': 1,
    'SF': 2,
    'PF': 3,
    'C': 4
}
data['Pos'] = data['Pos'].map(positions)

# Separate features (X) and target (y)
X = data.drop(['Player', 'Tm', 'Pos', 'Age', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'PF', 'G', 'GS', 'MP', 'TRB'], axis=1)
y = data['Pos']

# Task 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Task 1:")

linear_svc = LinearSVC(C=0.1, random_state=None, max_iter=5000)
linear_svc.fit(X_train_scaled, y_train)

print("Linear SVC Classifier Train set score: {:.3f}".format(linear_svc.score(X_train_scaled, y_train)))
print("Linear SVC Classifier Test set score: {:.3f}".format(linear_svc.score(X_test_scaled, y_test)))


# Task 2: Print confusion matrix
y_pred = linear_svc.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nTask 2:")
print("Confusion Matrix:")
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

# Task 3: 10-fold stratified cross-validation
cv_scores = cross_val_score(linear_svc, X, y, cv=10)
average_accuracy = cv_scores.mean()

print("\nTask 3:")
print("Accuracy of each fold:", cv_scores)
print("Average Accuracy across all folds:", average_accuracy)
