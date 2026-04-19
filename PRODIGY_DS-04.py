# Sentiment Analysis of Social Media Data

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Social media comments dataset
data = {
    "comments": [
        "I love this brand",
        "This product is amazing",
        "Very bad service",
        "I am not happy with this product",
        "Excellent quality",
        "Worst experience ever",
        "I really like it",
        "This is terrible",
        "Good product",
        "Not worth the money"
    ]
}

df = pd.DataFrame(data)

# Function to detect sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["comments"].apply(get_sentiment)

print(df)

# Count sentiment types
sentiment_count = df["Sentiment"].value_counts()

# Plot graph
plt.bar(sentiment_count.index, sentiment_count.values)
plt.title("Sentiment Analysis of Social Media Comments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.show()
