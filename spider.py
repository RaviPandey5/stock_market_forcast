import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from datetime import datetime, date, timedelta
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly

# Sidebar input configuration
st.sidebar.title("Stock Analysis Dashboard")
st.sidebar.markdown("**Analyze stock performance and forecast future prices.**")

user_input = st.sidebar.text_input('Enter Stock Ticker', "TSLA")
n_years = st.sidebar.slider('Years of Data:', 1, 10)
button_clicked = st.sidebar.button("GO")
period = n_years * 365
start = date.today() - timedelta(days=period)
end = datetime.today().strftime('%Y-%m-%d')

# Download stock data
@st.cache
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    return data

df = load_data(user_input, start, end)

# Tabs for different sections
ma, fundamental, news, predict = st.tabs(['📈 Moving Average', '📊 Fundamental Data', '📰 News', '🔮 Predict Price'])

# Moving Average Analysis Tab
with ma:
    st.title('Moving Average Trend')
    st.subheader(f'Data from {start.strftime("%Y")} to {end}')

    # Check if data is available
    if df.empty:
        st.error("No data available for the selected stock and time range.")
    else:
        st.write(df.describe())

        # Closing Price vs Time Chart
        st.subheader('Closing Price vs Time Chart')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        fig.update_layout(
            title=f'{user_input} Stock Closing Price',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Closing Price with 100-Day Moving Average
        st.subheader('Closing Price vs Time Chart with 100-Day MA')
        ma100 = df['Close'].rolling(100).mean().dropna()
        fig.add_trace(go.Scatter(x=ma100.index, y=ma100, mode='lines', name='100-Day MA'))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Closing Price with 100-Day and 200-Day Moving Averages
        st.subheader('Closing Price vs Time Chart with 100-Day & 200-Day MA')
        ma200 = df['Close'].rolling(200).mean().dropna()
        fig.add_trace(go.Scatter(x=ma200.index, y=ma200, mode='lines', name='200-Day MA'))
        st.plotly_chart(fig, use_container_width=True)

# Fundamental Data Tab
with fundamental:
    st.subheader(f"Daily **Closing Price** for {user_input}")
    if not df.empty:
        st.line_chart(df['Close'], height=250, use_container_width=True)
    else:
        st.write("No data available for the selected stock.")

# News Tab (Assuming you have the code to retrieve and display news here)
with news:
    st.write("Here you would display news related to the stock.")

# Forecasting with Prophet
def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    df = stock_data[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return df

def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Prediction Tab
with predict:
    st.header('Stock Price Prediction')
    col1, col2 = st.columns(2)
    ticker_symbol = col1.text_input('Enter Ticker Symbol', user_input)
    start_date = col2.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
    end_date = col2.date_input('End Date', value=pd.to_datetime('today'))
    forecast_horizon = col1.selectbox('Forecast Horizon', options=['1 year', '2 years', '3 years', '5 years'],
                                      format_func=lambda x: x.capitalize())
    
    horizon_mapping = {'1 year': 365, '2 years': 730, '3 years': 1095, '5 years': 1825}
    forecast_days = horizon_mapping[forecast_horizon]

    if st.button('Forecast Stock Prices'):
        with st.spinner('Fetching data...'):
            df_forecast = fetch_stock_data(ticker_symbol, start_date, end_date)
            if df_forecast.empty:
                st.error("No data available for the selected stock and time range.")
            else:
                with st.spinner('Training model...'):
                    model = train_prophet_model(df_forecast)
                    forecast = make_forecast(model, forecast_days)

                # Show forecast data
                st.subheader('Forecast Data')
                forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={'ds': 'Date', 'yhat': 'Predicted', 'yhat_lower': 'Lower Limit', 'yhat_upper': 'Upper Limit'})
                st.write(forecast_table.head())

                # Forecast Plot
                st.subheader('Forecast Plot')
                fig1 = plot_plotly(model, forecast)
                fig1.update_layout(
                    title=f'Forecast for {ticker_symbol}',
                    xaxis_title='Date',
                    yaxis_title='Predicted Price',
                    template="plotly_dark"
                )
                st.plotly_chart(fig1, use_container_width=True)
