import streamlit as st
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')


pickle_in = open("C:\\Users\\15623\\Downloads\\data.pkl", "rb")
data = pickle.load(pickle_in)


def predict_arima_food(start,end):
    rollingMean = data['FOODS'].rolling(window=30, center=False).mean()
    rollingStd = data['FOODS'].rolling(window=30, center=False).std()
    food_Series = data['FOODS']
    food_Series.fillna(food_Series.mean(), inplace=True)
    movingAverage = food_Series.rolling(window=30).mean()
    movingSTD = food_Series.rolling(window=30).std()
    foodSeriesDiff = food_Series - movingAverage
    foodSeriesDiff.fillna(foodSeriesDiff.mean(), inplace=True)

    model_foods = ARIMA(foodSeriesDiff, order=(2, 0, 2))
    results_ARIMA_foods = model_foods.fit()
    pred_foods = results_ARIMA_foods.get_prediction()
    y_forecasted_foods = pred_foods.predicted_mean
    y_truth_foods = foodSeriesDiff
    mse = ((y_forecasted_foods - y_truth_foods) ** 2).mean()
    mse_foods = np.sqrt(mse)
    rmse_foods = np.sqrt(mse_foods)
    f = plot_predict(results_ARIMA_foods, int(start), int(end), dynamic=False)
    
    st.pyplot(f)

    return rmse_foods


def predict_arima_household(start,end):
    household_Series = data['HOUSEHOLD']
    household_Series.fillna(household_Series.mean(), inplace=True)
    movingAverage = household_Series.rolling(window=30).mean().round()
    movingSTD = household_Series.rolling(window=30).std().round()
    householdSeriesDiff = household_Series - movingAverage
    householdSeriesDiff.fillna(householdSeriesDiff.mean(), inplace=True)

    model_household = ARIMA(householdSeriesDiff, order=(2, 0, 2))
    results_ARIMA_household = model_household.fit()

    pred_household = results_ARIMA_household.get_prediction()
    y_forecasted_household = pred_household.predicted_mean
    y_truth_household = householdSeriesDiff
    mse = ((y_forecasted_household - y_truth_household) ** 2).mean()
    mse_household = np.sqrt(mse)
    rmse_household = np.sqrt(mse_household)
    f = plot_predict(results_ARIMA_household, int(start), int(end), dynamic=False)
    st.pyplot(f)

    return rmse_household

def predict_arima_hobbies(start,end):
    hobbies_Series = data['HOUSEHOLD']
    hobbies_Series.fillna(hobbies_Series.mean(), inplace=True)
    movingAverage = hobbies_Series.rolling(window=30).mean().round()
    movingSTD = hobbies_Series.rolling(window=30).std().round()
    hobbiesSeriesDiff = hobbies_Series - movingAverage
    hobbiesSeriesDiff.fillna(hobbiesSeriesDiff.mean(), inplace=True)

    model_hobbies = ARIMA(hobbiesSeriesDiff, order=(2, 0, 2))
    results_ARIMA_hobbies = model_hobbies.fit()

    pred_hobbies = results_ARIMA_hobbies.get_prediction()
    y_forecasted_hobbies = pred_hobbies.predicted_mean
    y_truth_hobbies = hobbiesSeriesDiff
    mse = ((y_forecasted_hobbies - y_truth_hobbies) ** 2).mean()
    mse_hobbies = np.sqrt(mse)
    rmse_hobbies = np.sqrt(mse_hobbies)
    f = plot_predict(results_ARIMA_hobbies, int(start), int(end), dynamic=False)
    st.pyplot(f)

    return rmse_hobbies

def main():
    st.header("Arima")
    st.sidebar.title("What to do")
    option = st.sidebar.selectbox(
        'Choose the data series for which you want the prediction',
        ('Arima_food', 'Arima_household', 'Arima_hobbies'))

    if option == "Arima_food":
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h2 style="color:white;text-align:center;">Arima for Food category </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        start_day = st.text_input("start_date")
        end_day = st.text_input("end_date")
        result1 = ""

        if st.button("Predict"):
            result1 = predict_arima_food(start_day,end_day)
            st.success('The Root Mean Squared Error of our forecasts for FOODS Category is : {} '.format(result1))

    if option == "Arima_household":
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h2 style="color:white;text-align:center;">Arima for Household category </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        start_day = st.text_input("start_date")
        end_day = st.text_input("end_date")
        result2 = ""

        if st.button("Predict"):
            result2 = predict_arima_household(start_day,end_day)
            st.success('The Root Mean Squared Error of our forecasts for HOUSEHOLD Category is : {} '.format(result2))

    if option == "Arima_hobbies":
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h2 style="color:white;text-align:center;">Arima for hobbies category </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        start_day = st.text_input("start_date")
        end_day = st.text_input("end_date")
        result3 = ""

        if st.button("Predict"):
            result3 = predict_arima_hobbies(start_day,end_day)
            st.success('The Root Mean Squared Error of our forecasts for HOBBIES Category is : {} '.format(result3))

if __name__ == '__main__':
    main()
