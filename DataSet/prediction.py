import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class SpamClassifier:
    def __init__(self, model_path, data_path, num_words=5000, max_len=100):
        self.model_path = model_path
        self.data_path = data_path
        self.num_words = num_words
        self.max_len = max_len

        self.model = load_model(self.model_path)
        df = pd.read_csv(self.data_path)
        texts = df['text'].astype(str).values

        self.tokenizer = Tokenizer(num_words=self.num_words)
        self.tokenizer.fit_on_texts(texts)

    def preprocess_text(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len)
        return padded

    def predict(self, text):
        processed = self.preprocess_text(text)
        pred = self.model.predict(processed)[0]
        label = "Spam" if pred >= 0.5 else "Not Spam"
        return label, pred
