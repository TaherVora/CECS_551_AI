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
