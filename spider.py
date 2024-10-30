import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from datetime import datetime, date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from textblob import TextBlob
from newsapi import NewsApiClient

# Sidebar configuration
st.sidebar.title("Stock Analysis Dashboard")
st.sidebar.markdown("**Analyze stock performance and forecast future prices.**")

user_input = st.sidebar.text_input('Enter Stock Ticker', "TSLA")
n_years = st.sidebar.slider('Years of Data:', 1, 10)
button_clicked = st.sidebar.button("GO")
period = n_years * 365
start = date.today() - timedelta(days=period)
end = datetime.today().strftime('%Y-%m-%d')

# Download stock data with caching and error handling
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    # Check if data is empty or doesn't contain the 'Close' column
    if data.empty or 'Close' not in data.columns:
        return None
    # Ensure 'Close' column is numeric
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])  # Drop rows where 'Close' is NaN
    return data

# Load data and handle if None is returned
df = load_data(user_input, start, end)
if df is None:
    st.error(f"Error fetching data for ticker {user_input}. Please check the ticker symbol and try again.")

# Define tabs
ma, fundamental, news, predict = st.tabs(['📈 Moving Average', '📊 Fundamental Data', '📰 News', '🔮 Predict Price'])

# Moving Average Tab
with ma:
    st.title('Moving Average Trend')
    st.subheader(f'Data from {start.strftime("%Y")} to {end}')

    if df is None or df.empty:
        st.warning("No data available for the selected stock and time range.")
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

        # 100-Day Moving Average
        st.subheader('Closing Price with 100-Day MA')
        ma100 = df['Close'].rolling(100).mean().dropna()
        fig.add_trace(go.Scatter(x=ma100.index, y=ma100, mode='lines', name='100-Day MA'))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # 100-Day and 200-Day Moving Averages
        st.subheader('Closing Price with 100-Day & 200-Day MA')
        ma200 = df['Close'].rolling(200).mean().dropna()
        fig.add_trace(go.Scatter(x=ma200.index, y=ma200, mode='lines', name='200-Day MA'))
        st.plotly_chart(fig, use_container_width=True)

# Fundamental Data Tab
with fundamental:
    st.subheader(f"Daily **closing price** for {user_input}")
    if df is not None and not df.empty:
        st.line_chart(df['Close'], height=250, use_container_width=True)
    else:
        st.write("No data available for the selected stock.")

    # Additional data and information display
    stock_data = yf.Ticker(user_input)
    st.subheader("Additional Information")
    
    # Display stock actions
    actions = st.sidebar.checkbox("Stock Actions")
    if actions:
        st.write("Stock Actions:")
        st.write(stock_data.actions)
        
    # Display quarterly financials
    financials = st.sidebar.checkbox("Quarterly Financials")
    if financials:
        st.write("Quarterly Financials:")
        st.write(stock_data.quarterly_financials)
        
    # Display institutional shareholders
    shareholders = st.sidebar.checkbox("Institutional Shareholders")
    if shareholders:
        st.write("Institutional Shareholders:")
        st.write(stock_data.institutional_holders)
        
    # Display quarterly balance sheet
    balance_sheet = st.sidebar.checkbox("Quarterly Balance Sheet")
    if balance_sheet:
        st.write("Quarterly Balance Sheet:")
        st.write(stock_data.quarterly_balance_sheet)
        
    # Display quarterly cashflow
    cashflow = st.sidebar.checkbox("Quarterly Cashflow")
    if cashflow:
        st.write("Quarterly Cashflow:")
        st.write(stock_data.quarterly_cashflow)
        
    # Display quarterly earnings
    earnings = st.sidebar.checkbox("Quarterly Earnings")
    if earnings:
        st.write("Quarterly Earnings:")
        st.write(stock_data.quarterly_earnings)

# News Tab
with news:
    st.header(f'Latest News for {user_input}')

    # Initialize NewsAPI client
    newsapi = NewsApiClient(api_key='5ca85b2af4fb4e3d80d193cfd5a03e6a')  # Replace with your NewsAPI key

    # Fetch stock-related news using NewsAPI
    articles = newsapi.get_everything(
        q=user_input,
        language='en',
        sort_by='relevancy',
        page_size=10
    )['articles']

    # Display each article
    for i, article in enumerate(articles):
        title = article.get('title', 'No Title Available')
        publisher = article.get('source', {}).get('name', 'Unknown Publisher')
        publish_time = article.get('publishedAt', 'Unknown Date')
        summary = article.get('description', 'No summary available for this news article.')
        link = article.get('url', '#')

        with st.expander(f"News {i + 1}: {title}"):
            st.write(f"**Published on:** {publisher} - {publish_time}")
            st.write(f"**Summary:** {summary}")

            # Sentiment Analysis
            blob = TextBlob(summary)
            sentiment_score = blob.sentiment.polarity
            sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
            sentiment_color = "green" if sentiment_label == "Positive" else "red" if sentiment_label == "Negative" else "gray"
            st.markdown(f'<p><b>Sentiment:</b> <span style="color:{sentiment_color}">{sentiment_label}</span></p>', unsafe_allow_html=True)
            st.markdown(f"[Read Full Article]({link})", unsafe_allow_html=True)

# Prediction Tab
with predict:
    st.header('Stock Price Prediction')
    col1, col2 = st.columns(2)
    ticker_symbol = col1.text_input('Enter Ticker Symbol', user_input)
    start_date = col2.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
    end_date = col2.date_input('End Date', value=pd.to_datetime('today'))
    forecast_horizon = col1.selectbox('Forecast Horizon', options=['1 year', '2 years', '3 years', '5 years'],
                                      format_func=lambda x: x.capitalize())

    # Define forecast periods
    horizon_mapping = {'1 year': 365, '2 years': 730, '3 years': 1095, '5 years': 1825}
    forecast_days = horizon_mapping[forecast_horizon]

    # Fetch stock data for prediction
    def fetch_stock_data(ticker_symbol, start_date, end_date):
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        df = stock_data[['Adj Close']].reset_index()
        df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['y'])
        return df

    # Train Prophet model
    def train_prophet_model(df):
        model = Prophet()
        model.fit(df)
        return model

    # Make forecast
    def make_forecast(model, periods):
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast

    if st.button('Forecast Stock Prices'):
        with st.spinner('Fetching data...'):
            df_forecast = fetch_stock_data(ticker_symbol, start_date, end_date)
            if df_forecast.empty:
                st.error("No data available for the selected stock and time range.")
            else:
                with st.spinner('Training model...'):
                    model = train_prophet_model(df_forecast)
                    forecast = make_forecast(model, forecast_days)

                # Display forecast data
                st.subheader('Forecast Data')
                forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={'ds': 'Date', 'yhat': 'Predicted', 'yhat_lower': 'Lower Limit', 'yhat_upper': 'Upper Limit'})
                st.write(forecast_table.head())

                # Display forecast plot
                st.subheader('Forecast Plot')
                fig1 = plot_plotly(model, forecast)
                fig1.update_layout(
                    title=f'Forecast for {ticker_symbol}',
                    xaxis_title='Date',
                    yaxis_title='Predicted Price',
                    template="plotly_dark"
                )
                st.plotly_chart(fig1, use_container_width=True)
