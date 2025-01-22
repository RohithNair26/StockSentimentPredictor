StockSentimentPredictor
Overview
StockSentimentPredictor is a data-driven project that leverages sentiment analysis and topic modeling on Twitter data to understand and predict stock market trends. This repository contains the code and methodology to analyze public sentiment and its correlation with stock price movements for the top 25 stock tickers listed on Yahoo Finance.

Features
Sentiment analysis of tweets using TextBlob for polarity scoring.
Topic modeling to identify key themes in tweets.
Visualizations of sentiment trends, tweet distributions, and stock price movements.
Cleaned and preprocessed data for reproducible results.
Stock price trend analysis using 7-day moving averages.
Dataset
The dataset includes:

Tweets from top 25 Yahoo Finance stock tickers (30-09-2021 to 30-09-2022).
Stock market price and volume data for corresponding dates.
Columns:
Date: Timestamp of the tweet.
Tweet: Content of the tweet.
Stock Name: Stock ticker (e.g., TSLA, AAPL).
Company Name: Full name of the company.
Methodology
1. Data Preprocessing
Removal of URLs, special characters, and stopwords.
Conversion of text to lowercase for uniformity.
2. Sentiment Analysis
Technique: TextBlob for polarity scoring.
Polarity Scores: Categorized as positive, neutral, or negative sentiments.
3. Topic Modeling
Technique: Latent Dirichlet Allocation (LDA) for theme extraction.
Insights: Identifies trending topics and themes in the tweets.
4. Stock Price Analysis
Calculation of 7-day moving averages.
Correlation analysis between sentiment scores and stock price changes.
Installation
Prerequisites
Python 3.7+
Libraries: pandas, matplotlib, seaborn, textblob, re, warnings
Steps
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/StockSentimentPredictor.git
cd StockSentimentPredictor
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Place the datasets (stock_tweets.csv and stock_yfinance_data.csv) in the working directory.
Usage
Run the Analysis
Open the notebook or script.
Follow the comments in the code to:
Preprocess data.
Conduct sentiment analysis and topic modeling.
Visualize stock trends and sentiment correlations.
Output visualizations will be saved in the results folder.
Visualizations
Distribution of tweets per stock ticker.
Sentiment trends over time.
Stock price movement with sentiment correlation.
Contributing
Contributions are welcome! Please follow the steps below:

Fork the repository.
Create a new branch:
bash
Copy
Edit
git checkout -b feature-name
Commit your changes:
bash
Copy
Edit
git commit -m "Add feature-name"
Push to your branch:
bash
Copy
Edit
git push origin feature-name
Submit a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or suggestions:

Email: rohithnair2604@gmail.com.com
GitHub: RohithNair26
