# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:52:54 2024

@author: lenovo
"""

import joblib
import os
from flask import Flask, request, render_template
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Uncomment these lines if you need to download NLTK resources
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

app = Flask(__name__)

# Load the model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model = joblib.load(f)
    
    with open('TfidfVectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = joblib.load(f)

    print("Model and vectorizer loaded successfully!")

except Exception as e:
    model, tfidf_vectorizer = None, None
    print(f"Error during loading: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template("input.html")

@app.route('/submit', methods=["POST"])
def submit():
    try:
        # Read the user input
        text = request.form['userInput']
        if not text:
            return render_template('output.html', prediction_text="Input text is empty. Please enter some text.")
        
        # Clean the input text
        text = re.sub('[^a-zA-Z0-9]+', " ", text)
        
        # Tokenization, stopword removal, and lemmatization
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        preprocess_text = ' '.join(lemmatized_tokens)
        
        # Transform the preprocessed text using the TF-IDF vectorizer
        text_vectorized = tfidf_vectorizer.transform([preprocess_text])
        
        # Print to check if vectorization is successful
        print("Vectorized text shape:", text_vectorized.shape)
        
        # Make predictions using the model
        prediction = model.predict(text_vectorized)[0]
        
        # Map prediction to a label
        label = "Positive" if prediction == 1 else "Negative"
        
        return render_template('output.html', prediction_text=f'After Analysis, State of Mind was found: {label}')
    
    except AttributeError as e:
        print("AttributeError:", e)
        return render_template('output.html', prediction_text="AttributeError: The model or vectorizer is not properly loaded.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('output.html', prediction_text="An unexpected error occurred. Please check the console for more details.")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
