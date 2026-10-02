# import joblib
# import pandas as pd
# import os
# import re
# from urllib.parse import urlparse

# def extract_features_v1(url):
#     """Features for Model 1 (Structural - 19 features)"""
#     return {
#         'url_length': len(url),
#         'n_dots': url.count('.'),
#         'n_hypens': url.count('-'),
#         'n_underline': url.count('_'),
#         'n_slash': url.count('/'),
#         'n_questionmark': url.count('?'),
#         'n_equal': url.count('='),
#         'n_at': url.count('@'),
#         'n_and': url.count('&'),
#         'n_exclamation': url.count('!'),
#         'n_space': url.count(' ') + url.count('%20'),
#         'n_tilde': url.count('~'),
#         'n_comma': url.count(','),
#         'n_plus': url.count('+'),
#         'n_asterisk': url.count('*'),
#         'n_hastag': url.count('#'),
#         'n_dollar': url.count('$'),
#         'n_percent': url.count('%'),
#         'n_redirection': 0
#     }

# def extract_features_v2_v3(url):
#     """Features for Model 2 & 3 (Advanced - 31 features)"""
#     parsed = urlparse(url)
#     host = parsed.netloc
#     path = parsed.path
    
#     def count_digits(s):
#         return sum(c.isdigit() for c in s)

#     features = {
#         'length_url': len(url),
#         'nb_dots': url.count('.'),
#         'nb_hyphens': url.count('-'),
#         'nb_at': url.count('@'),
#         'nb_qm': url.count('?'),
#         'nb_and': url.count('&'),
#         'nb_or': url.count('|'),
#         'nb_eq': url.count('='),
#         'nb_underscore': url.count('_'),
#         'nb_tilde': url.count('~'),
#         'nb_percent': url.count('%'),
#         'nb_slash': url.count('/'),
#         'nb_star': url.count('*'),
#         'nb_colon': url.count(':'),
#         'nb_comma': url.count(','),
#         'nb_semicolumn': url.count(';'),
#         'nb_dollar': url.count('$'),
#         'nb_space': url.count(' ') + url.count('%20'),
#         'nb_www': 1 if 'www' in url.lower() else 0,
#         'nb_com': url.lower().count('.com'),
#         'nb_dslash': url.count('//'),
#         'http_in_path': 1 if 'http' in path.lower() else 0,
#         'https_token': 1 if 'https' in url.lower() else 0,
#         'ratio_digits_url': count_digits(url) / len(url) if len(url) > 0 else 0,
#         'ratio_digits_host': count_digits(host) / len(host) if len(host) > 0 else 0,
#         'punycode': 1 if 'xn--' in url.lower() else 0,
#         'port': 1 if parsed.port else 0,
#         'nb_subdomains': host.count('.') if host else 0,
#         'prefix_suffix': 1 if '-' in host else 0,
#         'random_domain': 0,
#         'shortening_service': 1 if any(s in host for s in ['bit.ly', 'goo.gl', 't.co', 'tinyurl']) else 0
#     }
#     return features

# def detect_phishing_ensemble_3(url, res_dir='res'):
#     m1_path = os.path.join(res_dir, 'model.pkl')
#     m2_path = os.path.join(res_dir, 'model_2.pkl')
#     m3_path = os.path.join(res_dir, 'model_3.pkl')
    
#     if not all(os.path.exists(p) for p in [m1_path, m2_path, m3_path]):
#         print("Error: Models not found. Run training scripts first.")
#         return

#     # Load models
#     model1 = joblib.load(m1_path)
#     model2 = joblib.load(m2_path)
#     model3 = joblib.load(m3_path)
    
#     # Extract features
#     fv1 = pd.DataFrame([extract_features_v1(url)])
#     fv_adv = extract_features_v2_v3(url)
    
#     # Align features for Model 2
#     m2_features = joblib.load(os.path.join(res_dir, 'model_2_features.pkl'))
#     fv2 = pd.DataFrame([fv_adv])[m2_features]

#     # Align features for Model 3
#     m3_features = joblib.load(os.path.join(res_dir, 'model_3_features.pkl'))
#     fv3 = pd.DataFrame([fv_adv])[m3_features]

#     # Predict Probabilities
#     p1 = model1.predict_proba(fv1)[0]
#     p2 = model2.predict_proba(fv2)[0]
#     p3 = model3.predict_proba(fv3)[0]
    
#     # Simple average ensemble
#     avg_prob_phish = (p1[1] + p2[1] + p3[1]) / 3
    
#     result = "PHISHING" if avg_prob_phish > 0.5 else "LEGITIMATE"
#     confidence = avg_prob_phish if result == "PHISHING" else (1 - avg_prob_phish)
    
#     print(f"\n--- 3-Model Ensemble Analysis for: {url} ---")
#     print(f"Model 1 (Structural) Phish Prob: {p1[1]*100:.2f}%")
#     print(f"Model 2 (Advanced) Phish Prob:   {p2[1]*100:.2f}%")
#     print(f"Model 3 (Massive) Phish Prob:    {p3[1]*100:.2f}%")
#     print(f"Ensemble Verdict: {result}")
#     print(f"Combined Confidence: {confidence*100:.2f}%")

# # if __name__ == "__main__":
# #     print("Phishing URL Detector (3-Model Ensemble)")
# #     while True:
# #         target_url = input("\nEnter a URL to check (or type 'quit' to exit): ").strip()
# #         if target_url.lower() == 'quit':
# #             break
# #         if target_url:
# #             if not target_url.startswith(('http://', 'https://')):
# #                 # Detect the context of plain domains
# #                 target_url = 'http://' + target_url
# #             detect_phishing_ensemble_3(target_url)



import joblib
import pandas as pd
import os
from .gemini_helper import explain_url_result
from .feature_extract import extract_features_v1, extract_features_v2_v3

def detect_phishing_ensemble_3(url):
    # Get current directory of the file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(current_dir, "res")

    m1_path = os.path.join(res_dir, "model.pkl")
    m2_path = os.path.join(res_dir, "model_2.pkl")
    m3_path = os.path.join(res_dir, "model_3.pkl")

    if not all(os.path.exists(p) for p in [m1_path, m2_path, m3_path]):
        raise FileNotFoundError("Models not found in " + res_dir)

    model1 = joblib.load(m1_path)
    model2 = joblib.load(m2_path)
    model3 = joblib.load(m3_path)

    fv1 = pd.DataFrame([extract_features_v1(url)])
    fv_adv = extract_features_v2_v3(url)

    # Load feature lists from disk
    m2_features = joblib.load(os.path.join(res_dir, "model_2_features.pkl"))
    m3_features = joblib.load(os.path.join(res_dir, "model_3_features.pkl"))

    fv2 = pd.DataFrame([fv_adv])[m2_features]
    fv3 = pd.DataFrame([fv_adv])[m3_features]


    p1 = model1.predict_proba(fv1)[0]
    p2 = model2.predict_proba(fv2)[0]
    p3 = model3.predict_proba(fv3)[0]

    avg_prob_phish = (p1[1] + p2[1] + p3[1]) / 3

    result = "PHISHING" if avg_prob_phish > 0.5 else "LEGITIMATE"
    confidence = avg_prob_phish if result == "PHISHING" else (1 - avg_prob_phish)

    model_probs = {
        "m1": p1[1],
        "m2": p2[1],
        "m3": p3[1]
    }

    # 🔥 Gemini explanation
    reason = explain_url_result(
        url=url,
        result=result,
        confidence=confidence,
        model_probs=model_probs
    )

    return {
        "result": result,
        "confidence": round(confidence * 100, 2),
        "reason": reason
    }
