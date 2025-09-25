Real-Time News Sentiment Dashboard

This project is a real-time news sentiment classification and visualization dashboard. It fetches the latest news headlines from NewsAPI (with GNews fallback), classifies each headline as Positive or Negative, and visualizes the results in a Streamlit-based dashboard.

Features

Fetch real-time news from NewsAPI or GNews.

Classify headlines as Positive or Negative using TextBlob.

Store predictions in Parquet format for history tracking.

Visualization capabilities include:

Display of latest headlines with sentiment labels.

Sentiment distribution using bar charts.

Positive sentiment trends over time using line charts.

Fully deployable on Streamlit Cloud.

Requirements

Python 3.9+ with the following dependencies:

streamlit
pandas
plotly
textblob
gnews
requests
pyarrow
