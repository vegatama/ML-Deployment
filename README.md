# ML-Deployment
# Use Case
#### Use Case Summary
#### Objective Statement:
* To find out how the highest and lowest scores obtained by students, as well as the longest and shortest hours of study done by students.
* To know relation between number of study hours and marks of the student.
* To predict student scores using machine learning linear regression.
* To get MAPE, R-Squared, MAE, and RMSE of simplel linear model.
* To deploy the model using MLflow.

#### Challenges:
* Small size of dataset

#### Methodology / Analytic Technique:
* Exploratory analysis (Graph Analysis)
* Simple Linear Model
* Model tracking using MLflow
* Deployment

#### Business Benefit:
* Know how to increase student grades.

#### Expected Outcome:
* Find out how the highest and lowest scores obtained by students, as well as the longest and shortest hours of study done by students
* Know the relation between number of study hours and marks of the student.
* Predict student scores using machine learning linear regression.
* Know the MAPE, R-Squared, MAE, and RMSE of the model.
* The model is deployed using MLflow.

# Business Understanding
Scoring in education is the process of applying standardized measurements for varying degrees of achievement in one course.</br>
This case requires data-driven answers to the following questions:
* Which are the highest and lowest scores obtained by students, as well as the longest and shortest hours of study done by students?
* How is relation between number of study hours and marks of the student?
* How is the prediction of student scores using machine learning linear regression?
* How much is the MAPE, R-Squared, MAE, and RMSE of the model?
* How to deploy the model using mlflow?

# Data Understanding
* Data of the number of hours student studied and the marks they got.
* The dataset has 2 columns and 25 rows.

#### Data Source
* Source Data: https://www.kaggle.com/code/turhancankargin/simple-linear-regression/data 

#### Data Dictionary 
* Hours 	: Number of study hours
* Scores	: Marks of the student

# Data preparation
Code Used :
* Python Version :Python 3.8.8
* Packages : Pandas, Numpy, Matplotlib, Seaborn, SKlearn, Warnings, MLflow, Logging, Urllib 

# Data Cleansing
The data is already clean, no missing value and mismatched data type, so we don't need to clean it.

## Exploratory Data Analysis
* Which are the highest and lowest scores obtained by students, as well as the longest and shortest hours of study done by students?

![image](https://user-images.githubusercontent.com/83635356/202833628-b69ea52e-7e9a-4067-815c-920706249273.png)

Based on this description, the highest student score is 9.2, while the lowest is 1.1. For the maximum length of study hours is 9.5 hours, while the shortest is 17 hours.

* How is relation between number of study hours and marks of the student?

![image](https://user-images.githubusercontent.com/83635356/202833728-8fd68f7c-0781-492f-a3d2-b6fa5f8f43fc.png)

Based on the graph above, it can be seen that study hours and student grades are directly proportional, which means that the longer a student studies, the higher the grade he will get. This proves that there is a strong influence of study hours on student scores.

# Feature Engineering

![image](https://user-images.githubusercontent.com/90773766/202834876-e296c923-628d-440b-b2d0-bf372abb269d.png)

After checking the distribution of hours and scores data with distplot and check the mean, median and mode. These columns have a nearly normal distribution, so they do not need to be scaled.

# Preprocessing Modeling
In Preprocessing Modeling We define X and y. and then, we split the train and test data with a test size of 1/3 of the total data. also, Get shape of the train and test data to know how many rows and columns we use.

# Modeling
We use Linear Regression as model. Linear regression is a type of supervised learning algorithm, commonly used for predictive analysis. linear regression is a predictive modeling technique. It is used whenever there is a linear relation between the dependent and the independent variables.

![image](https://user-images.githubusercontent.com/90773766/202835051-34148568-535d-4ee4-8e00-30ac2a41ffd5.png)

Based on the graph above, the red dot has a position close to the line, therefore the prediction results have a value that is close to the actual value.

# Evaluate Modeling
- **Root Mean Squared Error (RMSE)** is the square root of the mean squared error between the predicted and actual values.

With an RMSE of 7.29/0.09 we have a pretty decent model. because in order for us to know how good our model is, if we look at the RMSE value, we will normalize the RMSE with the distance we want to predict, and it can be seen if the result is closer to 0 than to 1, which means that a value closer to 0 represents a better fitting model.

- **Mean Absolute Error** calculates the average difference between the calculated values and actual values.

the MAE value is 6.27, which means that the average predicted value will decrease by 6.27 and if you look at the range of data we want to predict, it is between 14-97 which has changed slightly

- **MAPE** is a percentage error metric where the value corresponds to the average amount of error that predictions have. 

**MAPE value of 18% means it's low, but acceptable accuracy**. A MAPE less than 5% is considered as an indication that the forecast is acceptably accurate. A MAPE greater than 10% but less than 25% indicates low, but acceptable accuracy.

- Coefficient of determination also called as **R2** score is used to evaluate the performance of a linear regression model. 

**0.90 value of r2 score means a strong correlation**. The higher the R-Squared value the better. An R-squared value of above 0.75 (which is our r-squared score is 0.90) would be considered a strong correlation.

# Deployment
we use mlflow to deploy the model to the localhost. then we use mlflow ui to get this result;

![image](https://user-images.githubusercontent.com/90773766/202835082-ca6e9e63-b80d-4bd8-84ea-cdfd43b087b8.png)

# Result
Based on the graph above, study hours and student scores are directly proportional, which means that the longer a student studies, the higher the score he will get. This proves that there is a strong relationship between study hours and student scores.

In this dataset, it can be described that there are 25 study hours data with an average of 5.01. The minimum study hours is 1.1 and the maximum is 9.2, also there are 25 student scored data with an average of 51.4. The minimum student score is 17 and the maximum is 95.

**RMSE value obtained is 7.29** with the **normalized RMSE value being 0.09** and it can be seen that the result is closer to 0 than 1, which means that a value closer to 0 indicates a **more suitable model**. The **MAE** value is 6.27, which means that the average predicted value will decrease by 6.27 and if you look at the range of data we want to predict, it is between 14-97 which has changed slightly. The **MAPE value obtained is 18%**, which means that the **accuracy is acceptable even though it is low** because MAPE is greater than 10% but less than 25% indicates low but acceptable accuracy. The **R2 value obtained is 0.**9 which means it has a **strong correlation** because in general, the higher the R2, the better the model fits your data.

# Recommendation 
**Students with low scores are advised to study longer** than usual so that their grades are higher considering there is a strong relationship between study hours and student scores. Meanwhile, **students with high scores are also advised to study harder** in order to get higher scores or maintain their scores.

Since the average student studying for 5 hours gets a score of 51, students who only study for 5 hours are advised to increase their study time in order to get a good grade. Students who study for 7.4 hours are only able to get a score of 75 which means students also have to study for more than 7.4 hours so that their grades are better.

# Deployment Result
**MLflow** is an open source framework that makes it easy to track the machine learning model that was trained and the parameters, data, and metrics associated with that model.

**MLflow model tracking** can train different machine learning models and then make predictions with them in turn using a standard model prediction interface. In addition, MLflow model tracking can also register models in the MLflow model registry and track which models are used in production so that this information is easily accessible to everyone who works with those models.

Based on the MLflow model tracking, the **RMSE value obtained is 7.29** with the **normalized RMSE value being 0.09** and it can be seen that the result is closer to 0 than 1, which means that a value closer to 0 indicates a **more suitable model**. The **MAE value obtained is 6.27**, meaning that the average absolute difference between the actual value and the predicted value is 6.27 and is still **relatively good**. The **MAPE value obtained is 18%**, which means that the **accuracy is acceptable even though it is low** because MAPE is greater than 10% but less than 25% indicates low but acceptable accuracy. The **R2 value obtained is 0.**9 which means it has a **strong correlation** because in general, the higher the R2, the better the model fits your data.
