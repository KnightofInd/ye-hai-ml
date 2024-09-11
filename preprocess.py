from asyncio import subprocess
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
    processed_tweet = preprocess(text)  # Preprocess your tweet
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

# Example
custom_tweet = "they are stuck in tsunami can you help them i am giving this image of them"
result = classify_tweet_multi_output(custom_tweet)
print("The tweet belongs to:")
print("Target:", result['target'])

print("Event Type:", result['event_type'])
print("Label:", result['label'])