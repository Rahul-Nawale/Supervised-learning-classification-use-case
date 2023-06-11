"""
************************************************************************************************
** COMP534 Applied AI                                                                         **
** NAME: RAHUL NAWALE                                                                         **
** STUDENT ID - 201669264                                                                     **
** TASK - Assignment 1 - Supervised learning methods for solving a classification problem     **
************************************************************************************************
"""

# Import the pandas library
import pandas as pd

# To Train the model using KNN and the training dataset.
from sklearn.neighbors import KNeighborsClassifier
# Import cross-validation to evaluate the performance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score

# Import the Logistic Regression class from the sklearn.linear_model module
from sklearn.linear_model import LogisticRegression

# Import the Random Forest class from the sklearn.linear_model module
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Import the matplotlib.pyplot and seaborn libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import the train_test_split function from the sklearn.model_selection module
from sklearn.model_selection import train_test_split


# Load the dataset
df = pd.read_csv('dataset_assignment1.csv')

# Print the data information
print(f"\nData Information:\n{'-' * 50}")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")
print(f"Column Names: {', '.join(df.columns)}")
print(f"Data Types:\n{df.dtypes}\n")
print(f"\n{'-' * 50}")
print(f"Summary Statistics:\n{df.describe()}\n")
print("####################################################################################")

# Print out the number of samples for each class in the dataset
print("\nNumber of Samples per Class:\n", '-' * 30)
print(df.groupby('class').size().reset_index(name='count'))

# Plot histogram for each feature
ax = df.hist(figsize=(10, 15))
for i, axi in enumerate(ax.ravel()):
    axi.set_ylabel('Frequency')
plt.suptitle("Histogram for each feature of the dataset")
plt.show()

# Plot density plot for each feature
ax = df.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, figsize=(15, 15))
for i, axi in enumerate(ax.ravel()):
    axi.set_ylabel('Density')
plt.suptitle("Density plot for each feature of the dataset")
plt.show()

# Plot boxplot for each feature
ax = df.plot(kind='box', subplots=True, layout=(3, 4), sharex=False, sharey=False, figsize=(15, 10))
for i, axi in enumerate(ax.ravel()):
    axi.set_ylabel('Value')
plt.suptitle("Boxplot for each feature of the dataset")
plt.show()

print("\n####################################################################################")

# Print out the statistical description of features for each class
print("\nStatistical Description of Features per Class:\n", '-' * 50)
df_stats = df.groupby(['class']).describe().stack(level=0).reset_index()
df_stats.columns = ['class', 'features', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
df_stats = df_stats[['class', 'features', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
print(df_stats)
print("####################################################################################")

# Split data into input and output variables
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model using the training dataset
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Use K-Fold Cross Validation to tune the hyperparameters of the algorithm
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Plot the accuracy scores for different values of k
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title("Plot the accuracy scores for different values of k")
plt.show()

print("####################################################################################")

# Test the model on the testing dataset
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_KNN = knn.predict(X_test)

conf_mat_KNN = confusion_matrix(y_test, y_pred_KNN)
sns.heatmap(conf_mat_KNN, annot=True, cmap='Blues')
plt.title("Confusion matrix for KNN")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([0.5, 1.5], ["Negative", "Positive"])
plt.yticks([0.5, 1.5], ["Negative", "Positive"])
plt.show()

# Print the confusion matrix for KNN Classifier Model
print("\nConfusion Matrix for KNN Classifier Model:\n", '-' * 50)
print(confusion_matrix(y_test, y_pred_KNN))

# Pick up a model that gives you the best result on the validation dataset
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_KNN = knn.predict(X_test)

# Print the classification report for KNN Classifier Model
print("\nClassification Report for KNN Classifier Model:\n", '-' * 50)
print(classification_report(y_test, y_pred_KNN, target_names=['class 0', 'class 1']))

# Fit the KNN Classifier Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_KNN = knn.predict(X_test)

# Print the KNN Classifier Model
print("\nKNN Classifier Model:\n", '-' * 50)
print("Accuracy:", accuracy_score(y_test, y_pred_KNN))
print("Precision:", precision_score(y_test, y_pred_KNN))
print("Recall:", recall_score(y_test, y_pred_KNN))
print("F1 Score:", f1_score(y_test, y_pred_KNN))

print("\n", "#" * 60)


# Create a LogisticRegression object with a random state of 0
classifier = LogisticRegression(random_state=0)

# Train the logistic regression model using the training data and labels
classifier.fit(X_train, y_train)

# Use cross-validation to evaluate the performance of the logistic regression model
accuracies_LR = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)

# Print the mean accuracy and standard deviation of the logistic regression model
print(f"\n{'-' * 50}")
print("Logistic Regression Model Accuracy: {:.2f} %".format(accuracies_LR.mean() * 100))
print(f"\n{'-' * 50}")
print("Logistic Regression Model Standard Deviation: {:.2f} %".format(accuracies_LR.std() * 100))

# Use the logistic regression model to make predictions on the test data
y_pred_LR = classifier.predict(X_test)

# Plot the confusion matrix for the logistic regression model
conf_mat_LR = confusion_matrix(y_test, y_pred_LR)
sns.heatmap(conf_mat_LR, annot=True, cmap='Blues')
plt.title("Confusion matrix for logistic regression")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([0.5, 1.5], ["Negative", "Positive"])
plt.yticks([0.5, 1.5], ["Negative", "Positive"])
plt.show()


# Print the confusion matrix for the logistic regression model
print(f"\n{'-' * 50}")
print("Confusion Matrix of Logistic Regression Model:\n", conf_mat_LR)

# Generate a report of the precision, recall, and F1 score for the logistic regression model
print(f"\n{'-' * 50}")
print("Classification Report of Logistic Regression Model:\n", classification_report(y_test, y_pred_LR))

# Calculate the accuracy, precision, recall, and F1 score for the logistic regression model
print("\nLogistic Regression Model:\n", '-' * 50)
print("Accuracy:", accuracy_score(y_test, y_pred_LR))
print("Precision:", precision_score(y_test, y_pred_LR))
print("Recall:", recall_score(y_test, y_pred_LR))
print("F1 Score:", f1_score(y_test, y_pred_LR))

print("\n", "#" * 60)


# Create a Random Forest Classifier object with a random state of 0
classifier = RandomForestClassifier(random_state=0)

# Train the random forest model using the training data and labels
classifier.fit(X_train, y_train)

# Use cross-validation to evaluate the performance of the random forest model
accuracies_RF = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)

# Print the mean accuracy and standard deviation of the random forest model
print(f"\n{'-' * 50}")
print("Random Forest Model Accuracy: {:.2f} %".format(accuracies_RF.mean() * 100))
print(f"\n{'-' * 50}")
print("Random Forest Model Standard Deviation: {:.2f} %".format(accuracies_RF.std() * 100))

# Use the random forest model to make predictions on the test data
y_pred_RF = classifier.predict(X_test)

# Plot the confusion matrix for the random forest model
conf_mat_RF = confusion_matrix(y_test, y_pred_RF)
sns.heatmap(conf_mat_RF, annot=True, cmap='Blues')
plt.title("Confusion matrix for random forest")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([0.5, 1.5], ["Negative", "Positive"])
plt.yticks([0.5, 1.5], ["Negative", "Positive"])
plt.show()

# Print the confusion matrix for the random forest model
print(f"\n{'-' * 50}")
print("Confusion Matrix of Random Forest Model:\n", conf_mat_RF)

# Generate a report of the precision, recall, and F1 score for the random forest model
print(f"\n{'-' * 50}")
print("Classification Report of Random Forest Model:\n", classification_report(y_test, y_pred_RF))

# Calculate the accuracy, precision, recall, and F1 score for the random forest model
print("\nRandom Forest Model:\n", '-' * 50)
print("Accuracy:", accuracy_score(y_test, y_pred_RF))
print("Precision:", precision_score(y_test, y_pred_RF))
print("Recall:", recall_score(y_test, y_pred_RF))
print("F1 Score:", f1_score(y_test, y_pred_RF))

print("\n", "#" * 60)
