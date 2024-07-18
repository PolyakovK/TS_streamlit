import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Заголовок
st.title('Временные ряды. Аналитика и предсказания')

# Загрузка файла
uploaded_file = st.file_uploader("Загрузи свой дата фрейм", type=["xls"])
use_default = st.checkbox('Или используй дефолтный датасет')

if uploaded_file or use_default:
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df[df['Category'] == 'Furniture']
    else:
        df = pd.read_excel('Sample - Superstore.xls')
        df = df[df['Category'] == 'Furniture']

    df = pd.DataFrame(df.groupby('Order Date')['Sales'].sum().reset_index())
    df.set_index('Order Date', inplace=True)
    df = df.resample('W').sum()

    st.write("### Data")
    st.write(df)


    # Скользящая средняя
    n_window = 1
    moving_average_pred = df['Sales'].rolling(window=n_window, closed='left').mean()

    # Взвешенная скользящая средняя
    def weighted_moving_average(x, n, weights):
        weights = np.array(weights)
        wmas = x.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
        result = pd.Series(wmas, index=df.index).shift(1)
        return result

    n_window = 52
    weights = [0.6] * 1 + [0.1 / 48] * 48 + [0.3 / 3] * 3
    weighted_moving_average_pred = weighted_moving_average(df['Sales'], n=n_window, weights=weights)

    st.write("### Moving Averages")
    col1, col2 = st.columns(2)

    with col1:
        plt.figure(figsize=(16, 8))
        plt.plot(df['Sales'], linewidth=2, label='Actual', alpha=.7)
        plt.plot(moving_average_pred, linewidth=2, label='Moving Average', alpha=.7)
        plt.title('Moving Average')
        plt.legend()
        st.pyplot(plt)

    with col2:
        plt.figure(figsize=(16, 8))
        plt.plot(df['Sales'], linewidth=2, label='Actual', alpha=.7)
        plt.plot(weighted_moving_average_pred, linewidth=2, label=f'Weighted Moving Average', alpha=.7)
        plt.title('Weighted Moving Average')
        plt.legend()
        st.pyplot(plt)

    # ACF и PACF
    st.write("### ACF and PACF")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(df['Sales'], ax=ax1, lags=52)
    plot_pacf(df['Sales'], ax=ax2, lags=52)
    st.pyplot(fig)

    

    # Загрузка моделей
    sarimax_model = load('sarimax_model.joblib')
    prophet_model = load('prophet_model.joblib')
    catboost_model = load('catboost_model.joblib')

    # SARIMAX
    data_test = df[df['Sales'].index.year >= 2017]
    sarimax_pred = sarimax_model.forecast(len(data_test))
    sarimax_mae = mean_absolute_error(data_test, sarimax_pred)
    sarimax_r2 = r2_score(data_test, sarimax_pred)

    st.write("### SARIMAX")
    st.write(f"MAE = {sarimax_mae}")
    st.write(f"R2 = {sarimax_r2}")
    plt.figure(figsize=(20, 10))
    plt.plot(df['Sales'], label='Actual', marker='o')
    plt.plot(sarimax_pred, marker='v', color='g', label='Test Predict', linestyle=':')
    plt.legend()
    plt.title(f'SARIMAX, MAE = {sarimax_mae}')
    st.pyplot(plt)

    # Prophet
    data_prophet = df['Sales'].reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
    data_test = data_prophet[data_prophet['ds'].dt.year >= 2017]
    future = prophet_model.make_future_dataframe(periods=len(data_test), freq='W')
    forecast = prophet_model.predict(future)
    prophet_pred = forecast['yhat'][-len(data_test):]
    prophet_mae = mean_absolute_error(data_test['y'], prophet_pred)
    prophet_r2 = r2_score(data_test['y'], prophet_pred)

    st.write("### Prophet")
    st.write(f"MAE = {prophet_mae}")
    st.write(f"R2 = {prophet_r2}")
    plt.figure(figsize=(20, 10))
    plt.plot(df['Sales'], label='True Data', marker='o')
    plt.plot(forecast['ds'][-len(data_test):], prophet_pred, marker='v', linestyle=':', label=f'Forecast, MAE={prophet_mae}')
    plt.title(f'Prophet, MAE = {prophet_mae}')
    plt.legend()
    st.pyplot(plt)

    # CatBoost
    # Создаем фичи на основе лагов\
    data_train = df[df['Sales'].index.year < 2017]
    data_test = df[df['Sales'].index.year >= 2017]
    def create_lag_features(df, lags=[1, 2]):
        for lag in lags:
            df[f'lag_{lag}'] = df['Sales'].shift(lag)
        return df

    # Применяем функцию для создания фичей
    data_train = create_lag_features(data_train)
    data_test = create_lag_features(data_test)

    # Удаляем строки с NaN, появившиеся после создания лагов
    data_train = data_train.dropna()
    data_test = data_test.dropna()

    # Разделяем фичи и целевую переменную
    X_train = data_train.drop(columns=['Sales'])
    y_train = data_train['Sales']
    X_test = data_test.drop(columns=['Sales'])
    y_test = data_test['Sales']


    catboost_pred = catboost_model.predict(X_test)
    catboost_mae = mean_absolute_error(y_test, catboost_pred)
    catboost_r2 = r2_score(y_test, catboost_pred)

    st.write("### CatBoost")
    st.write(f"MAE = {catboost_mae}")
    st.write(f"R2 = {catboost_r2}")
    plt.figure(figsize=(10, 6))
    plt.plot(data_test.index, y_test, label='Actual Sales')
    plt.plot(data_test.index, catboost_pred, label='Predicted Sales', linestyle='--')
    plt.legend()
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    st.pyplot(plt)
