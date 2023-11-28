#pip install pandas nltk matplotlib


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
import matplotlib.pyplot as plt

def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalnum() and word not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

def analyze_feedback(data, top_n):
    # Combine all feedback into a single string
    all_feedback_text = ' '.join(data['feedback'])

    # Preprocess the text
    preprocessed_tokens = preprocess_text(all_feedback_text)

    # Calculate the frequency distribution of words
    freq_dist = FreqDist(preprocessed_tokens)

    # Display the top N most frequent words
    print(f"Top {top_n} Most Frequent Words:")
    for word, freq in freq_dist.most_common(top_n):
        print(f"{word}: {freq}")

    # Plot a bar graph for visualization
    plt.figure(figsize=(12, 6))
    freq_dist.plot(top_n, cumulative=False)
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    # Load the dataset from a CSV file
    dataset_path = 'data.csv'  # Replace with your actual file path
    feedback_data = pd.read_csv(dataset_path)

    # Get user input for the number of top words to analyze
    top_words_count = int(input("Enter the number of top words to analyze: "))

    # Analyze the feedback data
    analyze_feedback(feedback_data, top_words_count)
