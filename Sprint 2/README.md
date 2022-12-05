# Sprint 2: Machine Learning model

## Designing a machine learning model is an iterative process where we start with a baseline model and improve its performance by observing the performance metrics and the end goals remains optimizing the model. 
## In sprint 2 you will work on two different dataset and solve the retail problem statement using machine learning.
- Dataset_01: Regression, Classification 
- Dataset_02: Time Series

### CECS551_dataset_01

#### The data consists of 45 stores including store information and monthly sales. The objective is to predict sales of store in a week. As in dataset, size and time related data are given as feature, so analyze if sales are impacted by external factors, for example, how inclusion of holidays in a week soars the sales in store?
- Design a prediction model to forecast the weekly sales across first ten stores and use the same model to make predictions for store_11_35. There are external variables such as gas price, holidays, unemployment, and temperature for the given dataset. Evaluate the impact of these external variables on the accuracy of the model (do they help improve the accuracy?). Plot the relevant graphs. 
- Begin with Linear regression model to forecast the weekly sales using the given features.
- Create the following machine learning models: ARIMA, Ridge Regression and Boosting to predict sales.
- Communicate the model performance metrics and tabulate the comparison in report. Support your finding by validating the model accuracy across various stores (first 10 stores and store_11_35).

#### In stores.csv, we have a feature “type”, i.e., three types of stores – A (Super center), B (Discount center), and C (Neighborhood markets). Now, based on the given features in the dataset, can we predict the store type? Please consider only first 10 stores for this problem statement. 
- Consider the problem statement as multi-label classification problem. Use the below classification algorithms and perform hyper-parameter tuning for the Deep Learning models. 
--- Ensemble models (3 statistical methods)
--- Recurrent neural network (RNN)
--- Convolutional Neural Networks (CNN)
- Plot the relevant graphs and tabulate the performance metrics


### CECS551_dataset_02

#### Task 1: Design a machine learning model to make accurate predictions for product sales for next 10 days in advance (the data set includes daily unit sales per product) and compare the performance of different machine learning algorithms.

#### Background: The dataset also involves external variables, for exmaple, calendar-related information and selling prices. Thus, apart from the past unit sales of the products and the corresponding timestamps (e.g., date, weekday, week number, month, and year), there is also information available about: special events and holidays (e.g., Super Bowl, Valentine’s Day, and Orthodox Easter), organized into four classes, namely Sporting, Cultural, National, and Religious.

#### Selling prices, provided on a week-store level (average across seven days). If not available, this means that the product was not sold during the week examined. Although prices are constant on a weekly basis, they may change with time.
- Perform data preprocessing and exploratory data analysis. Perform down-casting (shrink dataset size)
- Feature engineering: create two new features using the information provided in Table 1. 
a) weather data
b) median income


Weather data: https://www.climate.gov/maps-data/dataset/past-weather-zip-code-data-table
Median income: https://data.census.gov/cedsci/table?q=ZCTA5%2090804%20Income%20and%20Poverty&tid=ACSST5Y2020.S1903


#### Use the below machine learning algorithms to model n-step ahead forecasting (n = 10).

- Begin with ARIMA and compare the RMSE values for each category. 
- Long short-term memory (LSTM) – Perform hyper-parameter to improve the model. 
- Plot the relevant graphs and tabulate the performance metrics of each model. 
