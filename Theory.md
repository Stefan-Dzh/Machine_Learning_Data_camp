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
print(predictions) "

There are two types of supervised learning—classification and regression. 
Binary classification is used to predict a target variable that has only two labels, typically represented numerically with a zero or a one.

Classifying labels of unseen data

1.Build a model
2.Model learns from the labeled data to the model as input
3.Pass unlabeled data to the model as input
4.Model predicts the labels of the unseen data

Labeled data = training data

FIRST MODEL
k-Nearest Neighbours
-Predict the label of a data point by
    Looking at the "k" closest labeled data points
    Taking a Majority Vote

Using scikit-learn to fit a classifier

"from sklearn.neighbours import KNeighborsClassifier
X = churn_df[["total_day_charge", "total_eve_charge"]].values #.values = goes to numpy arrays
y = churn_df['churn'].values
print(X.shape, y.shape)"
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X,y)

Predicting on unlabeled data

X_new = np.array([[56.8, 17.5],
                [24.4, 24.1],
                [50.1, 10.9]])

print(X_new.shape)
predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions)
