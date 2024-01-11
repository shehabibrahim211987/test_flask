# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model from test.py
from test import model, tokenizer, max_length

app = Flask(__name__)
CORS(app)

@app.route('/prediction')
def prediction():
	return render_template('prediction.html') # this should be under templates folder

@app.route('/prediction-api', methods=['POST'])
def prediction_api():
	test_sentence = request.form.get('test_sentence')
	sentence = test_sentence.lower()
	sequence = tokenizer.texts_to_sequences([sentence])
	test_sentence_padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
	output = model.predict(test_sentence_padded)
	return 'POSITIVE' if output[0][0] > 0.5 else 'NEGATIVE'

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)