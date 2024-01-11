import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
training_size_precent = 80
number_words = 10000
max_length = 100
MODEL_NAME = 'MOBILE_PHONE'

# Set input and output data
dataset = pd.read_csv('mobile_phones.csv')
sentences = []
labels = []
for index, record in dataset.iterrows():
	if (len(record.comment) > 5):
		sentences.append(record.comment.lower())
		labels.append(1 if record.rating >= 4 else 0)

# Split data into training and validation
training_size = round(len(sentences) * training_size_precent / 100)
print("Total sentences: " + str(len(sentences)) + "\nTraining Size: " + str(training_size))
training_sentences = sentences[0:training_size]
training_labels = labels[0:training_size]
testing_sentences = sentences[training_size:]
testing_labels = labels[training_size:]

# Tokenize
tokenizer = Tokenizer(num_words=number_words, oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)

# Test sentences
test_sentences = [
	'the exterior of the phone is very beautiful',
	'ang ganda naman',
	'this is not a good item',
	'may sira yung item, do not buy'
]

# Load Model
print(f'Loading model {MODEL_NAME}')

print('sulod')
model = load_model(f'{MODEL_NAME}.keras')
print(model.summary())
for sentence in test_sentences:
    sentence = sentence.lower()
    sequence = tokenizer.texts_to_sequences([sentence])
    test_sentence_padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    print(sentence)
    print(model.predict(test_sentence_padded))
    print("\n")