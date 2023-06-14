# Credit Risk Assessment 
![](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/Karthika/Images/credit-report.jpg)

## Project Background
Credit default risk is the chance that companies/individuals cannot make the required payments on their debt obligations, which can lead to a possibility of loss for a lender. Earlier credit analysts would perform risk management by analyzing the borrower’s credentials and capabilities, but this was prone to errors. With the advent of Machine learning, ML algorithms can perform a credit risk assessment with better precision and much faster than any humans.

## Project Goal
To quickly assess credit risk associated with companies or individuals with a high degree of accuracy.

## Project Objectives
1. Obtain source data and create a usable dataset for testing and training an ML model using logistic regression model and XGBoost model and compare the results.
2. Use a logistic regression model to analyze new source data.
3. Create a simple user interface for the ML model in AWS.

## Data Sets
* Credit Risk Dataset | Kaggle [https://www.kaggle.com/datasets/laotse/credit-risk-dataset]
* **[loan_data_2007_2014.csv](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/Resources/credit_risk_dataset.csv)**

## Tech-stack
Predict creditworthiness using Logistic Regression and XGBoost machine learning models on Jupyter Notebook and evaluate the model’s performance. 
Build an user interface for the models in Amazon Web Services using Lambda function.

## Data Pre-processing, Exploration and Analysis 
### Part 1 - Logistic Regression Model 

### Part 2 - Logistic Regression Model vs XGBoost Model
* Used Pandas `read_csv` function and Path module to read the "credit_risk_dataset.csv"
* Checked missing values by using `isnull().sum()` and used `dropna` to drop missing values
* Imported "plotly.express" and used `scatter_matrix` to determine the outliners
![](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/Images/scatter_matrix.png)
* Removed outliners `["person_age"]<=100]`, `["person_emp_length"]<=100]`,`["person_income"]<= 4000000]`
#### Given the nature of our dataset, we’d expect that we’re dealing with an imbalanced classification problem, meaning that we have considerably more non-default cases than default cases. Using the code below, we confirm that this is indeed the case with 78.3% of our dataset containing non-default cases.
* 78.3% = `credit_risk_df[credit_risk_df.loan_status == 0].loan_status.count()/credit_risk_df.loan_status.count()`
* Created the labels set (y) from the “loan_status” column
* One hot encoding of categorical variables, used `get_dummies`,and then create the features (X) DataFrame from the remaining columns.
*  Checked the balance of the labels variable (y) by using the `value_counts` function.
*  Splited the data into training and testing datasets by using `train_test_split`.
*  `!pip install xgboost` and fit a XG Boost model by using the training data (X_train and y_train)
*  Saved the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
*  Used `model_assess` function, evaluated the model’s performance and printed the classification report.
 ![](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/Images/lg_vs_XGB_classification_report.png)
### Analysis (Note: 0: healthy loan ; 1: high-risk loan)
* The overall accruacy for XGBoost is 94%, 13% higher than logistic regression model.
* XGBoost: The precision of 1(high-risk loan) is 95% vs 76% of 1 for LG model, recall for 1(high-risk loan) is 74%, higher than 1 in LG.
* Precision: Out of all portfolios that the model predicted would have high-risk loan, 94% did. It is a very good result.
* Recall: Out of all the portfolios that actually did have high risk loan, the model predicted this outcome correctly for 74% of high-risk-loans portfolios.

### Part 3 - Logistic Regression Model on new resource data
* Loaded all the appropriate libraries 
* Used Pandas `read_csv` function and Path module to read the "loan_data_2015.csv"
* Changed Loan status to binary code 0 - 1, Issued: 0, Current: 1
* Drop unnecessary columns
* Checked missing values by using `isnull().sum()` and used `dropna` to drop missing values
* Checked outlier scatterplot. 
* Created the labels set (y) from the “loan_status” column
* Splited the data into training and testing datasets by using `train_test_split`.
* Saved the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
* Used `model_assess` function, evaluated the model’s performance and printed the classification report.
 ![](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/c7ec0e84e11241779d432e61156da0058b59fc76/Images/Screenshot%202023-06-14%20190043.png)
### Analysis (Note: 0: healthy loan ; 1: high-risk loan)
* The recall for class 0.0 is 0.25, which means that the model only captures 25% of the actual positive instances of high-risk-loans portfolios.
* The F1-score for class 0.0 is 0.40, indicating a moderate balance between precision and recall for this class.
* the weighted average precision is 1.00, the weighted average recall is 0.25, and the weighted average F1-score is 0.40.
* Accuracy measures the overall correctness of the model's predictions. The accuracy reported in the classification report is 0.25, meaning that the model correctly predicted 25% of the instances in the dataset. Pot


## Challenges 
* AWS  dependency issues included numpy, pandas, and sklearn. Finding alternative libraries and keeping within size restrictions was not possible. Resolved by using a "next-best" alternative platform - flask. (


## Group Members
- Karthika Ramachandran
- Shan Lu
- Mohammad Zahur
- Larry Gagnon
