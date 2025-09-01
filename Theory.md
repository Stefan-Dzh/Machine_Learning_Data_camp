What is machine learning?

Machine learning is the process whereby:

- Computers are given the ability to learn to make decisions from data without being explicitly programmed!

Example:

Is an email spam or not?
Assigning new books to different categories.

Unsupervised learning
-Uncovering hidden patterns from unlabeled data
-Example:
Grouping customers into distinct categories.(Clustering)

Supervised learning
-The predicted values are known
-Aim: Predict the target values of unseen data, given the features

Types of Supervised learning
-Classification: Target variable consists of categories - binary(Yes/No)
-Regression - Target variable is continuous

Before you use supervised learning
-Requirements:
    No missing values
    Data in numeric format
    Data stored in pandas DataFrame or Numpy array
Perform Exploratory Data Analysis(EDA) first

scikit-learn syntax - workflow is repeatable
"from sklearn.module import Model
model = Model()
model.fit(X(features),y(target variable))
predictions = model.predict(X_new)
print(predictions)"