# Credit Risk Assessment 

## Project background
Credit default risk is the chance that companies/individuals cannot make the required payments on their debt obligations, which can lead to a possibility of loss for a lender. Earlier credit analysts would perform risk management by analyzing the borrower’s credentials and capabilities, but this was prone to errors. With the advent of Machine learning, ML algorithms can perform a credit risk assessment with better precision and much faster than any humans.

## Project Goal
To quickly assess credit risk associated to companies or individuals with a high degree of accuracy.

## Project Objectives
1.  Obtain source data and create a usable dataset for testing and training an ML model.
2.  Create an ML model that will produce a positive or negative credit risk rating for each company or individual
3.  Create a user friendly output (suggestion only).
4.  Create a simple user interface for the ML model (this is a suggestion only).

## Data sets
Credit Risk Dataset | Kaggle [https://www.kaggle.com/datasets/laotse/credit-risk-dataset]

## Tech-stack
To start this machine learning project, download the Credit Risk Dataset. Load the dataset into a data frame and remove rows of data NaN values. Also, convert the categorical values into numerical values using Label encoding. Our data is imbalanced. Hence, we use the stratifiedKFold method to split the dataset into training and testing sets.
Machine learning algorithms that can be used are KNN, logistic regression, and XGBoost(Extreme Gradient Boosting) algorithms. You can use the performance metrics like Accuracy, Precision, Recall, and F1 score to evaluate your model’s performance. However, since the training data was imbalanced, the Area Under the Curve for the ROC curve would be a better evaluation metric.

## Group members
- Karthika Ramachandran
- Shan Lu
- Mohammad Zahur
- Larry Gagnon
