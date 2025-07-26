# Chatbot Project

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Structure](#structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Prediction and Response](#prediction-and-response)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project is a simple chatbot that uses Natural Language Processing (NLP) and a neural network to understand and respond to user inputs. The chatbot is trained on a predefined set of intents and patterns and can respond to various types of queries based on those patterns.

## Features

- Handles basic greetings and farewells.
- Responds to questions about the bot's name.
- Provides information about shop items and availability.
- Displays shop hours.
- Checks stock availability.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/chatbot.git
   cd chatbot
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Usage

1. **Prepare the training data:**
   ```python
   python prepare_data.py
   ```

2. **Train the model:**
   ```python
   python train_model.py
   ```

3. **Run the chatbot:**
   ```python
   python chatbot.py
   ```

## Structure

- **intents.json**: Contains the predefined intents, patterns, and responses.
- **words.pkl**: Pickle file storing the vocabulary.
- **classes.pkl**: Pickle file storing the classes (intents).
- **chatbot_model.h5**: Trained model.
- **prepare_data.py**: Script for data preprocessing.
- **train_model.py**: Script for training the model.
- **chatbot.py**: Main script to run the chatbot.

## Data Preprocessing

1. **Tokenization and Lemmatization**: Tokenizes the patterns into words and lemmatizes them to reduce words to their base form.
   ```python
   from nltk.stem import WordNetLemmatizer
   import nltk

   lemmatizer = WordNetLemmatizer()
   words = []
   classes = []
   documents = []
   ignore_letters = ['?', '!', '.', ',']

   for intent in intents['intents']:
       for pattern in intent['patterns']:
           word_list = nltk.word_tokenize(pattern)
           words.extend(word_list)
           documents.append((word_list, intent['tag']))
           if intent['tag'] not in classes:
               classes.append(intent['tag'])

   words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
   words = sorted(set(words))
   classes = sorted(set(classes))
   ```

2. **Saving Preprocessed Data**: Stores the processed words and classes in pickle files.
   ```python
   import pickle
   pickle.dump(words, open('words.pkl', 'wb'))
   pickle.dump(classes, open('classes.pkl', 'wb'))
   ```

3. **Bag of Words**: Converts input sentences into a bag-of-words representation to use as input for the model.
   ```python
   def bag_of_words(sentence):
       sentence_words = clean_up_sentence(sentence)
       bag = [0] * len(words)
       for w in sentence_words:
           for i, word in enumerate(words):
               if word == w:
                   bag[i] = 1
       return np.array(bag)
   ```

## Model Training

1. **Preparing Training Data**: Creates training data by converting the documents into bag-of-words arrays and their corresponding intent classes into one-hot encoded arrays.
   ```python
   training = []
   output_empty = [0] * len(classes)

   for document in documents:
       bag = []
       word_patterns = document[0]
       word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
       for word in words:
           bag.append(1) if word in word_patterns else bag.append(0)
               
       output_row = list(output_empty)
       output_row[classes.index(document[1])] = 1
       training.append([bag, output_row])
       
   random.shuffle(training)
   training = np.array(training, dtype=object)

   train_x = np.array(training[:, 0].tolist())
   train_y = np.array(training[:, 1].tolist())
   ```

2. **Building the Neural Network**: Defines and trains a neural network using TensorFlow and Keras.
   ```python
   import tensorflow as tf

   model = tf.keras.Sequential()
   model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
   model.add(tf.keras.layers.Dropout(0.5))
   model.add(tf.keras.layers.Dense(64, activation='relu'))
   model.add(tf.keras.layers.Dropout(0.5))
   model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

   sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
   model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

   model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
   model.save('chatbot_model.h5')
   ```

## Prediction and Response

1. **Predicting the Intent**: Uses the trained model to predict the intent of a user input.
   ```python
   def predict_class(sentence):
       bow = bag_of_words(sentence)
       res = model.predict(np.array([bow]))[0]
       ERROR_THRESHOLD = 0.25
       result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
       result.sort(key=lambda x: x[1], reverse=True)
       return_list = []
       for r in result:
           return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
       return return_list
   ```

2. **Generating the Response**: Maps the predicted intent to a response from the intents JSON structure.
   ```python
   def get_response(intents_list, intents_json):
       tag = intents_list[0]['intent']
       list_of_intents = intents_json['intents']
       for i in list_of_intents:
           if i['tag'] == tag:
               result = random.choice(i['responses'])
               break
       return result
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any features, bug fixes, or improvements.



[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
