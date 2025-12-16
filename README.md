# Breast-Cancer-Prediction
This project builds a logistic regression model on the Breast Cancer Wisconsin(Diagnostic) dataset to predict whether a breast tumor is malignant or benign.

## Breast Cancer Prediction Using Logistic Regression
# Overview
This project focuses on building a breast cancer prediction system using Logistic Regression. The goal is to classify tumors as benign or malignant based on diagnostic features. The project demonstrates a complete end-to-end machine learning workflow, from data loading to model evaluation and prediction.
This project is beginner-friendly, well-structured, and suitable for showcasing core data science and machine learning skills.

## Dataset


Source: Breast Cancer Wisconsin (Diagnostic) Dataset


Target Variable: Diagnosis (Malignant / Benign)


Features: Numeric measurements related to cell nuclei characteristics such as radius, texture, perimeter, area, smoothness, etc.



**Tools & Technologies**


Programming Language: Python


Libraries Used:


NumPy


Pandas


Matplotlib


Seaborn


Scikit-learn





## Project Steps


**Data Loading**

Import the breast Cancer dataset from sklearn

Load the dataset into a pandas DataFrame with feature_names as columns


**Exploratory Data Analysis (EDA)**

 Display the first five rows of the dataset using .head()
 
 Add the target column to the DataFrame and display the last five rows using .tail()
 
 Analyze the dataset:
 
 Use .shape to check the number of rows and columns
 
 Use .info() for an overview of column types and non-null values
 
 Check for missing values using .isnull().sum()
 
 Display summary statistics with .describe()
 
 Analyze the target variable distribution using .value_counts()


**Data Preprocessing**

Separate the features (X) and target variable (Y):

X = data_frame.drop(columns='label', axis=1)

Y = data_frame['label']

**Splitting the Dataset**

Split the data into training and testing sets (80% training, 20% testing):
 
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

 **Model Training**
 
Train a logistic regression model:

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, Y_train)

 **Model Evaluation**
 
Evaluate the model using accuracy score:

On training data:

from sklearn.metrics import accuracy_score

X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy on training data = ', training_data_accuracy)

On testing data:

X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data = ', test_data_accuracy)

**Building a Predictive System**

Build a system to predict whether a tumor is benign or malignant based on new input data


## Results


The Logistic Regression model achieved high accuracy on the test dataset.


The model effectively distinguishes between malignant and benign tumors.


Logistic Regression proved to be a reliable and interpretable model for this classification task.



## How to Run


Clone this repository:
git clone <repository-url>



Navigate to the project directory:
cd breast-cancer-prediction



Install required dependencies:
pip install -r requirements.txt



## Run the notebook or script:
jupyter notebook

or
python breast_cancer_prediction.py




## Conclusion
This project demonstrates a complete machine learning pipeline using Logistic Regression for medical diagnosis. It highlights skills in data analysis, preprocessing, model training, and evaluation, making it a strong portfolio project for data science and machine learning roles.
