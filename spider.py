#
# from plotly import graph_objs as go
# from datetime import datetime, date, timedelta
# import streamlit as st
# import pandas as pd
# import numpy as np
# from prophet import Prophet
# import yfinance as yf
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from prophet.plot import plot_plotly
#
#
# ma, fundamental, news, predict = st.tabs(['Moving Average', 'Fundamental Data', 'News', 'Predict Price'])
# end = datetime.today().strftime('%Y-%m-%d')
# user_input = st.sidebar.text_input('Enter Stock Ticker', "TSLA")
# n_years = st.sidebar.slider('Years of Data:', 1, 10)
# button_clicked = st.sidebar.button("GO")
# period = n_years * 365
# start = date.today() - timedelta(days=period)
# start.strftime('%m%d%y')
# df = yf.download(user_input, start=start, end=end)
# a = start.strftime('%Y')
# b = end = datetime.today().strftime('%Y')
#
# with ma:
#
#     st.title('Moving Average Trend')
#     st.subheader('Data from ' + " " + str(a) + " to " + str(b))
#     st.write(df.describe())
#
#     st.subheader('Closing Price vs Time Chart')
#     fig = go.Figure()
#
#     fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
#
#     fig.update_layout(title=f'{user_input} Stock ', xaxis_title='Date', yaxis_title='Price',
#                       xaxis_rangeslider_visible=True)
#
#     st.plotly_chart(fig)
#
#
# def fundamental_data():
#     st.subheader("""Daily **closing price** for """ + user_input)
#
#
#
#
# def fetch_stock_data(ticker_symbol, start_date, end_date):
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
#     df = stock_data[['Adj Close']].reset_index()
#     df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
#     return df
#
# def train_prophet_model(df):
#     model = Prophet()
#     model.fit(df)
#     return model
#
# def make_forecast(model, periods):
#     future = model.make_future_dataframe(periods=periods)
#     forecast = model.predict(future)
#     return forecast
#
# def calculate_performance_metrics(actual, predicted):
#     mae = mean_absolute_error(actual, predicted)
#     mse = mean_squared_error(actual, predicted)
#     rmse = np.sqrt(mse)
#     return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
#
# def pred():
#     st.empty()
#     st.header('Stock Price Prediction')
#     st.subheader('User Input Parameters')
#     col1, col2 = st.columns(2)
#     ticker_symbol = col1.text_input('Enter Ticker Symbol', user_input)
#     start_date = col2.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
#     end_date = col2.date_input('End Date', value=pd.to_datetime('today'))
#     forecast_horizon = col1.selectbox('Forecast Horizon', options=['1 year', '2 years', '3 years', '5 years'],
#                                       format_func=lambda x: x.capitalize())
#     horizon_mapping = {'1 year': 365, '2 years': 730, '3 years': 1095, '5 years': 1825}
#     forecast_days = horizon_mapping[forecast_horizon]
#
#     if st.button('Forecast Stock Prices'):
#         with st.spinner('Fetching data...'):
#             df = fetch_stock_data(ticker_symbol, start_date, end_date)
#
#         with st.spinner('Training model...'):
#             model = train_prophet_model(df)
#             forecast = make_forecast(model, forecast_days)
#
#         # Show input parameters
#         st.subheader('User Input Parameters')
#         col3, col4 = st.columns(2)
#         col3.write(f'Ticker Symbol: {ticker_symbol}')
#         col3.write(f'Start Date: {start_date}')
#         col4.write(f'End Date: {end_date}')
#         col4.write(f'Forecast Horizon: {forecast_horizon}')
#
#         # Show forecast data
#         st.subheader('Forecast Data')
#         st.write(
#             'The table below shows the forecasted stock prices along with the lower and upper bounds of the predictions.')
#         forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
#             columns={'ds': 'Date', 'yhat': 'Predicted', 'yhat_lower': 'Lower Limit',
#                      'yhat_upper': 'Upper Limit'})
#         st.write(forecast_table.head())
#
#         st.subheader('Forecast Plot')
#         st.write('The plot below visualizes the predicted stock prices with their confidence intervals.')
#         fig1 = plot_plotly(model, forecast)
#         fig1.update_traces(marker=dict(color='red'), line=dict(color='white'))
#         st.plotly_chart(fig1)
#
#
# with predict:
#     pred()
# def local_css(file_name):
#     with open(file_name) as f:
#         st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#
#
# local_css("style.css")







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from datetime import datetime, date, timedelta
from stocknews import StockNews
from prophet import Prophet

from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly

ma, fundamental, news, predict = st.tabs(['Moving Average', 'Fundamental Data', 'News', 'Predict Price'])

# start = "20-01-01"
end = datetime.today().strftime('%Y-%m-%d')

user_input = st.sidebar.text_input('Enter Stock Ticker', "TSLA")

n_years = st.sidebar.slider('Years of Data:', 1, 10)
button_clicked = st.sidebar.button("GO")
period = n_years * 365
start = date.today() - timedelta(days=period)
start.strftime('%m%d%y')
df = yf.download(user_input, start=start, end=end)
# df = data.DataReader(user_input,'yahoo',start,end)
a = start.strftime('%Y')
b = end = datetime.today().strftime('%Y')

with ma:
    # Describing Data
    st.title('Moving Average Trend')
    st.subheader('Data from ' + " " + str(a) + " to " + str(b))
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time Chart')
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))

    fig.update_layout(title=f'{user_input} Stock ', xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma100, name='Close'))

    fig.update_layout(title=f'{user_input} Stock ', xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()

    fig.add_trace(go.Scatter(x=df.index, y=ma200, name='Close'))

    fig.update_layout(title=f'{user_input} Stock ', xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)


def fundamental_data():
    st.subheader("""Daily **closing price** for """ + user_input)
    # get data on searched ticker
    stock_data = yf.Ticker(user_input)
    # get historical data for searched ticker
    stock_df = stock_data.history(period='1d', start='2020-01-01', end=None)
    # print line chart with daily closing prices for searched ticker
    st.line_chart(stock_df.Close)

    st.subheader("""Last **closing price** for """ + user_input)
    # define variable today
    today = datetime.today().strftime('%Y-%m-%d')
    # get current date data for searched ticker
    stock_lastprice = stock_data.history(period='1d', start=today, end=today)
    # get current date closing price for searched ticker
    last_price = (stock_lastprice.Close)
    # if market is closed on current date print that there is no data available
    if last_price.empty == True:
        st.write("No data available at the moment")
    else:
        st.write(last_price)

    # get daily volume for searched ticker
    st.subheader("""Daily **volume** for """ + user_input)
    st.line_chart(stock_df.Volume)

    # additional information feature in sidebar
    st.sidebar.subheader("""Display Additional Information""")
    # checkbox to display stock actions for the searched ticker
    actions = st.sidebar.checkbox("Stock Actions")
    if actions:
        st.subheader("""Stock **actions** for """ + user_input)
        display_action = (stock_data.actions)
        if display_action.empty == True:
            st.write("No data available at the moment")
        else:
            st.write(display_action)

    # checkbox to display quarterly financials for the searched ticker
    financials = st.sidebar.checkbox("Quarterly Financials")
    if financials:
        st.subheader("""**Quarterly financials** for """ + user_input)
        display_financials = (stock_data.quarterly_financials)
        if display_financials.empty == True:
            st.write("No data available at the moment")
        else:
            st.write(display_financials)

    # checkbox to display list of institutional shareholders for searched ticker
    major_shareholders = st.sidebar.checkbox("Institutional Shareholders")
    if major_shareholders:
        st.subheader("""**Institutional investors** for """ + user_input)
        display_shareholders = (stock_data.institutional_holders)
        if display_shareholders.empty == True:
            st.write("No data available at the moment")
        else:
            st.write(display_shareholders)

    # checkbox to display quarterly balance sheet for searched ticker
    balance_sheet = st.sidebar.checkbox("Quarterly Balance Sheet")
    if balance_sheet:
        st.subheader("""**Quarterly balance sheet** for """ + user_input)
        display_balancesheet = (stock_data.quarterly_balance_sheet)
        if display_balancesheet.empty == True:
            st.write("No data available at the moment")
        else:
            st.write(display_balancesheet)

    # checkbox to display quarterly cashflow for searched ticker
    cashflow = st.sidebar.checkbox("Quarterly Cashflow")
    if cashflow:
        st.subheader("""**Quarterly cashflow** for """ + user_input)
        display_cashflow = (stock_data.quarterly_cashflow)
        if display_cashflow.empty == True:
            st.write("No data available at the moment")
        else:
            st.write(display_cashflow)

    # checkbox to display quarterly earnings for searched ticker
    earnings = st.sidebar.checkbox("Quarterly Earnings")
    if earnings:
        st.subheader("""**Quarterly earnings** for """ + user_input)
        display_earnings = (stock_data.financials)
        if display_earnings.empty == True:
            st.write("No data available at the moment")
        else:
            st.write(display_earnings)

    # checkbox to display list of analysts recommendation for searched ticker
    analyst_recommendation = st.sidebar.checkbox("Analysts Recommendation")
    if analyst_recommendation:
        st.subheader("""**Analysts recommendation** for """ + user_input)
        display_analyst_rec = (stock_data.recommendations)
        if display_analyst_rec.empty == True:
            st.write("No data available at the moment")
        else:
            st.write(display_analyst_rec)


with fundamental:
    fundamental_data()

import streamlit as st
from newsapi import NewsApiClient
from textblob import TextBlob
from datetime import datetime

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='5ca85b2af4fb4e3d80d193cfd5a03e6a')  # Replace with your NewsAPI key

# Fetch and display news articles
with news:
    st.header(f'Latest News for {user_input}')

    # Fetch stock-related news using NewsAPI
    articles = newsapi.get_everything(
        q=user_input,
        language='en',
        sort_by='relevancy',
        page_size=10  # Limit to 10 articles
    )['articles']

    # Display each article with interactive elements
    for i, article in enumerate(articles):
        title = article.get('title', 'No Title Available')
        publisher = article.get('source', {}).get('name', 'Unknown Publisher')
        publish_time = article.get('publishedAt', 'Unknown Date')
        summary = article.get('description', 'No summary available for this news article.')
        link = article.get('url', '#')

        # Convert publish time to a readable format if available
        if publish_time:
            publish_time = datetime.strptime(publish_time, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')

        with st.expander(f"News {i + 1}: {title}"):
            st.write(f"**Published on:** {publisher} - {publish_time}")
            st.write(f"**Summary:** {summary}")

            # Perform sentiment analysis on the summary
            blob = TextBlob(summary)
            sentiment_score = blob.sentiment.polarity
            sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
            sentiment_color = "green" if sentiment_label == "Positive" else "red" if sentiment_label == "Negative" else "gray"

            # Display sentiment with color
            st.markdown(
                f'<p><b>Sentiment:</b> <span style="color:{sentiment_color}">{sentiment_label}</span></p>',
                unsafe_allow_html=True
            )

            # Link to full article
            st.markdown(
                f"[Read Full Article]({link})",
                unsafe_allow_html=True
            )



import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly


# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    df = stock_data[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return df


# Function to train the Prophet model
def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model


# Function to make the forecast
def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


# Function to calculate performance metrics
def calculate_performance_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


# Streamlit app
def pred():
    st.empty()
    st.header('Stock Price Prediction')
    st.subheader('User Input Parameters')
    col1, col2 = st.columns(2)
    ticker_symbol = col1.text_input('Enter Ticker Symbol', user_input)
    start_date = col2.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
    end_date = col2.date_input('End Date', value=pd.to_datetime('today'))
    forecast_horizon = col1.selectbox('Forecast Horizon', options=['1 year', '2 years', '3 years', '5 years'],
                                      format_func=lambda x: x.capitalize())

    # Introduction
    # Set up the layout
    # Convert the selected horizon to days
    horizon_mapping = {'1 year': 365, '2 years': 730, '3 years': 1095, '5 years': 1825}
    forecast_days = horizon_mapping[forecast_horizon]

    if st.button('Forecast Stock Prices'):
        with st.spinner('Fetching data...'):
            df = fetch_stock_data(ticker_symbol, start_date, end_date)

        with st.spinner('Training model...'):
            model = train_prophet_model(df)
            forecast = make_forecast(model, forecast_days)

        # Show input parameters
        st.subheader('User Input Parameters')
        col3, col4 = st.columns(2)
        col3.write(f'Ticker Symbol: {ticker_symbol}')
        col3.write(f'Start Date: {start_date}')
        col4.write(f'End Date: {end_date}')
        col4.write(f'Forecast Horizon: {forecast_horizon}')

        # Show forecast data
        st.subheader('Forecast Data')
        st.write(
            'The table below shows the forecasted stock prices along with the lower and upper bounds of the predictions.')
        forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={'ds': 'Date', 'yhat': 'Predicted', 'yhat_lower': 'Lower Limit',
                     'yhat_upper': 'Upper Limit'})
        st.write(forecast_table.head())

        st.subheader('Forecast Plot')
        st.write('The plot below visualizes the predicted stock prices with their confidence intervals.')
        fig1 = plot_plotly(model, forecast)
        fig1.update_traces(marker=dict(color='red'), line=dict(color='white'))
        st.plotly_chart(fig1)


with predict:
    pred()

def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")

