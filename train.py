import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
word_index = tokenizer.word_index

# Pad input with zeroes at the end
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding='post', truncating='post')
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding='post', truncating='post')

# Convert to NP array
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# Create the model
model = tf.keras.Sequential([
	tf.keras.layers.Embedding(number_words, 16, input_length=max_length),
	tf.keras.layers.GlobalAveragePooling1D(),
	tf.keras.layers.Dense(24, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model
history = model.fit(
	training_padded,
	training_labels,
	epochs=25,
	validation_data=(testing_padded, testing_labels),
	verbose=2
)
model.save(f'{MODEL_NAME}.keras')

# Test the model
test_sentences = [
	'the exterior of the phone is very beautiful',
	'may sira yung item, do not buy'
]
sequences = tokenizer.texts_to_sequences(test_sentences)
test_sentences_padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
print(model.predict(test_sentences_padded))