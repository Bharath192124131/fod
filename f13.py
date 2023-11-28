# pip install pandas nltk matplotlib wordcloud

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Replace this with your actual dataset containing customer reviews
reviews_data = pd.DataFrame({
    'ReviewID': [1, 2, 3, 4, 5],
    'ReviewText': ["The product is excellent and worth every penny.",
                   "Not satisfied with the quality, very disappointed.",
                   "Amazing product, highly recommended!",
                   "Could be better. It didn't meet my expectations.",
                   "Great value for the price. I love it!"]
})

# Combine all reviews into a single string
all_reviews_text = ' '.join(reviews_data['ReviewText'])

# Tokenize the text into words
tokens = word_tokenize(all_reviews_text)

# Remove stopwords (common words that don't carry much meaning)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

# Calculate the frequency distribution of words
freq_dist = FreqDist(filtered_tokens)

# Display the top 10 most frequent words
print("Top 10 Most Frequent Words:")
print(freq_dist.most_common(10))

# Plot a word cloud for visualization
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
