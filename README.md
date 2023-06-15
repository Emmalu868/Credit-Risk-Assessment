# Credit Risk Assessment 
![](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/Images/credit-report.jpg)

## Group Members
- Karthika Ramachandran
- Shan Lu
- Mohammad Zahur
- Larry Gagnon

## About the project
Credit risk assessment is the process that lending companies use to evaluate the creditworthiness of customers before extending credit to protect their businesses from late/non payment. It involves analyzing various factors such as financial history, credit scores, income, assets, existing debts, and other relevant information to assess the risk associated with extending credit to an individual or entity. Earlier credit analysts would perform risk management by analyzing the borrower’s credentials and capabilities, but this was prone to errors. With the advent of Machine learning, ML algorithms can perform a credit risk assessment with better precision and much faster than any humans. <br>

This project aims at- <br>
* analysing the historical lending activity to build Logistic Regression and XGBoost models to predict the creditworthiness of borrowers
* evaluating the models' performance
* creating a simple user interface in Flask 

## Data Sets
* [credit_risk_dataset](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/Resources/credit_risk_dataset.csv) | [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
* [loan_data_2015](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/Resources/loan_data_2015.csv) | [Drop Box](https://www.dropbox.com/sh/7oslws1xhsm1zbf/AABc2smPMio5-_cQHLsrBT0Xa/Dataset?dl=0&subfolder_nav_tracking=1)

## Data Cleanup & Model Training and Evaluation 
### Part 1 - [Logistic Regression Model](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/LogisticRegression.ipynb) 
**Step 1: Data Exploration and Preprocessing** 
* Used Pandas `read_csv` function to read the "credit_risk_dataset.csv" file as a DataFrame. 
* Detected and removed null values using `dropna` function.
* Identified outliers using `scatterplot matrix` and removed them.
* Reviwed the data types and encoded categorical variables into numerical variables using `get_dummies` function.<br>

**Step 2: Split the Data into Training and Testing Sets**
* Created the labels set (y) from the “loan_status” column, and then created the features (X) DataFrame from the remaining columns.
* Checked the balance of the labels variable (y) by using the `value_counts` function.
* Split the data into training and testing datasets by using `train_test_split`.
* Resampled the data using the `RandomOverSampler` module from the `imbalanced-learn library`.<br>

**Step 3: Create a Logistic Regression Model**
*  Created a logistic regression model by using the resampled training data (`X_oversampled` and `y_oversampled`).
*  Saved the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.
*  Evaluated the model’s performance by calculating the accuracy score of the model, generating a confusion matrix and printing the classification report.
#### Analysis:
![](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/Images/2score.png) <br>
##### Accuracy Score 
The model has an accuracy score of 70% which is not a very good score for credit risk assessment where the consequences of misclassification can have significant financial implications.
##### Precision
* Precision for class 0 (non-defaulting borrowers): Out of all the instances predicted as non-defaulting borrowers, 92% of them are correctly classified. This indicates a low false positive rate for non-defaulting borrowers, which is desirable.
* Precision for class 1 (defaulting borrowers): Out of all the instances predicted as defaulting borrowers, only 40% of them are correctly classified. This suggests a higher false positive rate for defaulting borrowers, which means the model is less accurate in identifying the borrowers who are likely to default.<br>
While a high precision for the negative class (non-defaulting borrowers) is generally desirable to minimize false positives, a low precision for the positive class (defaulting borrowers) can be problematic. In credit risk assessment, accurately identifying defaulting borrowers is crucial to mitigate financial risks.
##### Recall
* Recall for class 0 (non-defaulting borrowers): Out of all the actual instances of non-defaulting borrowers, the model correctly identifies 67% of them. This means that the model has a moderate ability to capture non-defaulting borrowers.
* Recall for class 1 (defaulting borrowers): Out of all the actual instances of defaulting borrowers, the model correctly identifies 78% of them. This suggests that the model has a higher ability to identify defaulting borrowers compared to non-defaulting borrowers.<br>
A high recall for the positive class (defaulting borrowers) is desirable in credit risk assessment as it indicates that the model is effective at capturing a significant portion of borrowers who are likely to default. However, a lower recall for the negative class (non-defaulting borrowers) implies that the model may miss some non-defaulting borrowers, leading to false negatives.
##### F1 Score
* F1 score for class 0 (non-defaulting borrowers): The F1 score considers both precision and recall and provides a balanced evaluation metric. An F1 score of 77% for non-defaulting borrowers suggests that the model achieves a good balance between precision and recall for this class.
* F1 score for class 1 (defaulting borrowers): The F1 score of 53% for defaulting borrowers indicates that the model's performance in correctly identifying defaulting borrowers needs improvement, as it achieves a lower balance between precision and recall for this class.

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
*  Split the data into training and testing datasets by using `train_test_split`.
*  `!pip install xgboost` and fit a XG Boost model by using the training data (X_train and y_train)
*  Saved the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
*  Used `model_assess` function, evaluated the model’s performance and printed the classification report.
 ![](https://github.com/Emmalu868/Credit-Risk-Assessment/blob/main/Images/lg_vs_XGB_classification_report.png)
#### Analysis (Note: 0: healthy loan ; 1: high-risk loan)
* The overall accruacy for XGBoost is 94%, 13% higher than logistic regression model.
* XGBoost: The precision of 1(high-risk loan) is 95% vs 76% of 1 for LG model, recall for 1(high-risk loan) is 74%, higher than 1 in LG.
* Precision: Out of all portfolios that the model predicted would have high-risk loan, 95% did. It is a very good result.
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
* AWS dependency issues included numpy, pandas, and sklearn.
* Finding alternative libraries and keeping within size restrictions was not possible. Resolved by using a "next-best" alternative platform - flask.



