import json
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version warnings from sklearn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Define preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = ' '.join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)
    return text

# Load the saved models
model_target = joblib.load('model_target.sav')
model_event_type = joblib.load('model_event_type.sav')
model_label = joblib.load('model_label.sav')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.sav')

def classify_tweet_multi_output(text):
    processed_tweet = preprocess(text)  # Preprocess the tweet
    vec_tweet = vectorizer.transform([processed_tweet])
    
    # Predict for each model
    target_prediction = model_target.predict(vec_tweet)[0]
    event_type_prediction = model_event_type.predict(vec_tweet)[0]
    label_prediction = model_label.predict(vec_tweet)[0]
    
    return {
        'target': target_prediction,
        'event_type': event_type_prediction,
        'label': label_prediction
    }

# Load posts from the JSON file with utf-8 encoding and error handling
try:
    with open('kya yaar.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Successfully loaded JSON data!")
except FileNotFoundError:
    print("Error: The file 'posts.json' was not found.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    exit(1)
except UnicodeDecodeError as e:
    print(f"Error decoding file due to encoding issue: {e}")
    exit(1)

# Analyze each post assuming data is a list of posts
results = []
for post in data:  # Loop over the list directly
    text = post['content']  # Using the 'content' field from your JSON structure
    classification = classify_tweet_multi_output(text)
    
    # Gather relevant information
    result = {
        'content': text,
        'target': classification['target'],
        'event_type': classification['event_type'],
        'label': classification['label'],
        'date': post['date'],  # Include date if needed
        'like_count': post['like_count'],  # Include like count
        'retweet_count': post['retweet_count'],  # Include retweet count
        'hashtags': post['hashtags'],  # Include hashtags
        'media_url': post['media_url']  # Include media URL if available
    }
    results.append(result)

# Print out results
for result in results:
    print("Content:", result['content'])
    print("Target:", result['target'])
    print("Event Type:", result['event_type'])
    print("Label:", result['label'])
    print("Date:", result['date'])
    print("Likes:", result['like_count'])
    print("Retweets:", result['retweet_count'])
    print("Hashtags:", result['hashtags'])
    print("Media URL:", result['media_url'])
    print("\n")

# Optional: Save the results to a new JSON file
with open('classified_posts.json', 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile, indent=4)
