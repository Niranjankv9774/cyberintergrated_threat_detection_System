# 🛡️ Phishing URL Detection System

## 📖 About
The Phishing Detection module is a structural URL analysis engine designed to identify potentially malicious websites without needing to "browse" or download the page content. 

### How it Works: 3-Model Ensemble
The system uses an **Ensemble Learning** approach, combining three different machine learning models to provide a high-accuracy verdict:

1.  **Model 1 (Structural)**: Analyzes 19 basic features like URL length, special character counts (`@`, `?`, `-`), and redirection flags.
2.  **Model 2 (Advanced)**: Focuses on 31 advanced features including domain-to-path ratios, presence of brand tokens (`.com`, `www`), and shortening services (`bit.ly`, `goo.gl`).
3.  **Model 3 (Massive)**: A deep-learning-tuned model that processes features from a massive dataset to catch subtle patterns in TLD usage and punycode.

**Verdict**: The final result is an average probability. If the combined score is >50%, it is flagged as **PHISHING**.

### 🌲 Algorithm: Random Forest
All three models in the ensemble utilize the **Random Forest** algorithm (`RandomForestClassifier`). This algorithm was chosen for its:
- **Accuracy**: It effectively handles non-linear relationships between URL features.
- **Robustness**: By using an ensemble of multiple decision trees, it reduces false positives caused by unusual but legitimate URL structures.
- **Feature weighted**: It automatically weighs the importance of different URL components (like the presence of `@` vs. a `.com` suffix).

---

## 🚀 How to Use

### 1. Through the Mobile App
1.  Open the **Phishing Detection** page.
2.  Enter a URL into the input field.
3.  Click **Check URL**.
4.  The system will display the verdict, a confidence percentage, and an AI-generated reason for the result.

### 2. Input Best Practices
To get the most accurate results, use the "Formal Web Format":

| Input Type | Example | Recommendation |
| :--- | :--- | :--- |
| **Simple Domain** | `google.com` | ✅ Supported (System adds `http://`) |
| **Secure URL** | `https://www.google.com` | ⭐ **Best Accuracy** (Provides full trust tokens) |
| **IP Address** | `127.0.0.1:8000` | ✅ Supported |
| **File Paths** | `C:\test.html` | ❌ **Blocked** (The system will reject local paths) |

### 3. Understanding the "Reason"
The "Reason" section uses GenAI to explain *why* the ML model reached its conclusion. It might point out:
*   Missing security tokens (SSL/HTTPS).
*   Suspicious character density (too many dots or hyphens).
*   Typos in common domains (e.g., `.cpn` instead of `.com`).

---

## 🛠️ Developer Notes
- **Feature Extraction**: Logic is located in `myapp/feature_extract.py`.
- **Detection Core**: The ensemble logic is in `myapp/detect.py`.
- **Validation**: Input validation (regex) is handled in `myapp/views.py` to prevent malformed strings from reaching the model.
