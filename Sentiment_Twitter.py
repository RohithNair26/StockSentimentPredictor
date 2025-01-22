import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

tweets = pd.read_csv("D:/Trimester 4/TSMA/StockTweets/stock_tweets.csv")
returns = pd.read_csv("D:/Trimester 4/TSMA/StockTweets/stock_yfinance_data.csv")
tweets['Date'] = pd.to_datetime(tweets['Date'])
print (tweets.head())
returns['Date'] = pd.to_datetime(returns['Date'])
print (returns.head())

#distribution of Tweets
tweets['Stock Name'].unique()
count = tweets['Stock Name'].value_counts()
perc = count/count.sum()
dstr = pd.DataFrame({'Count':count, 'Percentage':perc}).reset_index()
dstr

def plot_company_stock(company, id, axs, stk):
    # Filter the stock data for the given company
    comp = stk[stk['Stock Name'] == company].copy()  # Avoid SettingWithCopyWarning
    comp['Date'] = pd.to_datetime(comp['Date'])  # Ensure Date is in datetime format
    comp['Moving_Avg'] = comp['Adj Close'].rolling(window=7).mean()  # 7-Day Moving Average
    
    # Plot Adjusted Close and Moving Average
    axs[id].plot(comp['Date'], comp['Moving_Avg'], label='7-Day Moving Average', color='orange')
    axs[id].scatter(comp['Date'], comp['Adj Close'], marker='.', color='skyblue', label='Adjusted Close')

    # Set title and labels
    axs[id].set_title(f'{company}')
    axs[id].set_xlabel('Date')
    axs[id].set_ylabel('Adj Close')
    
    # Rotate x-axis labels for better readability
    axs[id].tick_params(axis='x', rotation=90)

fig, axs = plt.subplots(5, 5, figsize=(15, 15))  # Create subplots
axs = axs.flatten()

# Loop through each company and plot its data
returns = pd.read_csv("D:/Trimester 4/TSMA/StockTweets/stock_yfinance_data.csv")
for i, company in enumerate(returns.sort_values(by='Stock Name')['Stock Name'].unique()):
    plot_company_stock(company, i, axs, returns)

plt.tight_layout()  # Adjust subplots to prevent overlap
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_company_tweets(company, id, axs, tweet_counts):
    # Filter the tweet counts data for the given company
    company_data = tweet_counts[tweet_counts['Stock Name'] == company]
    
    # Plot Tweet Counts for the company
    axs[id].scatter(company_data['Date'], company_data['Tweet Count'], marker='.', label=f'{company} Tweet Counts')
    
    
    # Set title and labels
    axs[id].set_title(f'{company}')
    axs[id].set_xlabel('Date')
    axs[id].set_ylabel('Tweet Count')
    
    # Rotate x-axis labels for better readability
    axs[id].tick_params(axis='x', rotation=90)


# Group by 'Stock Name' and 'Date' to get tweet counts
tweets = pd.read_csv("D:/Trimester 4/TSMA/StockTweets/stock_tweets.csv")
tweets['Date'] = pd.to_datetime(tweets['Date'], errors='coerce')
tweets = tweets.dropna(subset=['Date']) 
tweet_counts = tweets.groupby([tweets['Stock Name'], tweets['Date'].dt.date]).size().reset_index(name='Tweet Count')

# Get unique company names sorted by Stock Name
companies = tweet_counts.sort_values(by='Stock Name')['Stock Name'].unique()

# Create subplots for each company
fig, axs = plt.subplots(5, 5, figsize=(15, 15))
axs = axs.flatten()

# Loop through each company and plot its tweet counts
for i, company in enumerate(companies):
    plot_company_tweets(company, i, axs, tweet_counts)

plt.tight_layout()  # Adjust subplots to prevent overlap
plt.show()
# Define the sample size for each stock (e.g., 2 rows per stock)
sample_size_per_stock = 30  # Adjust this number based on your actual dataset size

# Sample the same number of rows for each stock
tweets_sampled = tweets.groupby('Stock Name').apply(lambda x: x.sample(n=sample_size_per_stock, random_state=42))

# Drop the extra index created by groupby
tweets_sampled = tweets_sampled.reset_index(drop=True)

# Display the sampled DataFrame
count = tweets_sampled['Stock Name'].value_counts()
perc = count/count.sum()
dstr = pd.DataFrame({'Count':count, 'Percentage':perc}).reset_index()
print (dstr)

import pandas as pd
import re

# Define a list of common stopwords (manually or use a library like sklearn)
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

# Preprocessing function to clean tweets
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove @mentions (usernames)
    text = re.sub(r"@\w+", "", text)
    # Remove any non-alphabetical characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert text to lowercase
    text = text.lower()
    # Split text into words
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    return " ".join(words)  # Return the cleaned text as a single string
tweets_sampled['cleaned_tweet'] = tweets_sampled['Tweet'].apply(preprocess_text)

from textblob import TextBlob

# Function to calculate the sentiment polarity score
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Sentiment score (-1 to 1)

# Apply the sentiment analysis function to the cleaned tweets
tweets_sampled['sentiment_score'] = tweets_sampled['cleaned_tweet'].apply(get_sentiment)
# Sort the DataFrame by sentiment_score
tweets_sampled_sorted = tweets_sampled.sort_values(by='sentiment_score')

# Sample from top (positive), middle (neutral), and bottom (negative)
positive_sample = tweets_sampled_sorted.tail(4)  # Last 4 rows (positive sentiment)
negative_sample = tweets_sampled_sorted.head(4)  # First 4 rows (negative sentiment)
neutral_sample = tweets_sampled_sorted.iloc[len(tweets_sampled_sorted) // 2 - 2: len(tweets_sampled_sorted) // 2 + 2]  # Middle 4 rows (neutral sentiment)

# Combine the samples into one DataFrame
sampled_entries = pd.concat([positive_sample, negative_sample, neutral_sample])

# Display the sampled entries
print(sampled_entries[['cleaned_tweet', 'sentiment_score']].head(12))

# Function to calculate sentiment counts
def sentiment_counts(stock_name, tweets_df):
    stock_tweets = tweets_df[tweets_df['Stock Name'] == stock_name]
    positive_count = (stock_tweets['sentiment_score'] > 0).sum()
    negative_count = (stock_tweets['sentiment_score'] < 0).sum()
    neutral_count = (stock_tweets['sentiment_score'] == 0).sum()
    return positive_count, negative_count, neutral_count

# Function to plot sentiment counts
def plot_sentiment_counts(stock_name, positive_count, negative_count, neutral_count):
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    sentiment_values = [positive_count, negative_count, neutral_count]
    
    plt.figure(figsize=(8, 5))
    plt.bar(sentiment_labels, sentiment_values, color=['green', 'red', 'gray'])
    plt.title(f'Sentiment Analysis for {stock_name}')
    plt.ylabel('Number of Tweets')
    plt.show()

# Example usage for a specific stock
stock_name = 'AAPL'  # Replace with the stock you want to analyze
positive_count, negative_count, neutral_count = sentiment_counts(stock_name, tweets_sampled)
plot_sentiment_counts(stock_name, positive_count, negative_count, neutral_count)