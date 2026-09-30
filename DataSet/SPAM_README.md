# 📧 Spam Email Detection System

## 📖 About
The Spam Detection module is a Natural Language Processing (NLP) engine designed to classify emails as either **Spam** or **Not Spam** (Ham) based on their textual content.

### How it Works: LSTM Neural Network
Unlike the Phishing Detection module (which uses Random Forest), the Spam Detector utilizes a **Deep Learning** approach:

1.  **Architecture**: It uses a **Long Short-Term Memory (LSTM)** network, which is a type of Recurrent Neural Network (RNN) specifically designed to understand context and long-range dependencies in text.
2.  **Preprocessing**:
    *   **Tokenization**: The email text is broken down into tokens, focusing on the top 5000 most frequent words.
    *   **Padding**: Every email is standardized to a fixed length of 100 words to ensure consistent input for the neural network.
3.  **Layers**:
    *   **Embedding Layer**: Converts words into dense vectors of fixed size (128).
    *   **LSTM Layer**: Processes the sequence of word vectors to capture the "meaning" and intent of the email.
    *   **Dropout Layer**: Prevents the model from "memorizing" specific emails (overfitting).
    *   **Dense (Sigmoid) Layer**: Outputs a probability between 0 and 1.

**Verdict**: If the probability is $\ge 0.5$, the email is classified as **Spam**.

---

## 🚀 How to Use

### 1. Through the Mobile App
1.  Navigate to the **Spam Email Detection** section.
2.  Paste the content of the email you wish to analyze into the text area.
3.  Click **Predict**.
4.  The system will display whether the email is "Spam" or "Not Spam".

### 2. Best Practices for Accuracy
*   **Paste Full Content**: Include the subject line and the main body for better context.
*   **Language**: The current model is optimized for English text.
*   **Length**: While the model can handle long emails, it focuses on the first 100 tokens. Ensure the most relevant parts of the email (like suspicious links or urgent requests) are included.

---

## 🛠️ Developer Notes
-   **Model File**: The trained model is saved as `result/spam_mail.h5`.
-   **Classifier**: The wrapper class for prediction is in `prediction.py`.
-   **Training**: The training logic, including the LSTM architecture, is defined in `training.py`.
-   **Database**: Reports are logged in the `spam_email_reports` table via `myapp/views.py`.
