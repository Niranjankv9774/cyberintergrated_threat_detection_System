import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"emails.csv")

# Data preprocessing
texts = df['text'].astype(str).values
labels = df['spam'].values

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)  # Use top 5000 words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to the same length
max_len = 100  # You can adjust the max length as needed
X = pad_sequences(sequences, maxlen=max_len)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))  # Removed input_length to avoid deprecation warning
model.add(LSTM(units=128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2000, batch_size=64, validation_data=(X_test, y_test))

# Ensure the save directory exists
save_dir = r"D:\RISS 2025\HG - BTech\web\cyberintegratedproject\result"
os.makedirs(save_dir, exist_ok=True)

# Save the model
model_save_path = os.path.join(save_dir, "spam_mail.h5")
model.save(model_save_path)

print(f"Model saved to {model_save_path}")
print(df.columns)
