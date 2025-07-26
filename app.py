from flask import Flask, request, jsonify, render_template
import json
import random
import pickle
import numpy as np
import tensorflow as tf
import os
from nltk.stem import WordNetLemmatizer

# Setup
lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHATBOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "prepared"))

# Load resources
intents_path = os.path.join(CHATBOT_DIR, 'intents.json')
words_path = os.path.join(CHATBOT_DIR, 'words.pkl')
classes_path = os.path.join(CHATBOT_DIR, 'classes.pkl')
model_path = os.path.join(CHATBOT_DIR, 'chatbot_model.h5')

intents_json = json.loads(open(intents_path).read())
words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))
model = tf.keras.models.load_model(model_path)

# Utility functions
def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]), verbose=0)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res[0]) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"

    tag = intents_list[0]['intent']
    intent_data = next((i for i in intents_json['intents'] if i['tag'] == tag), None)

    if intent_data and 'responses' in intent_data:
        return random.choice(intent_data['responses'])
    return "Sorry, I don't have an answer for that."

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_message = request.json.get("message")
    intents = predict_class(user_message)
    response = get_response(intents, intents_json)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
