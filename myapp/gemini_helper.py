import google.generativeai as genai

# Directly set API key (replace with your actual key)
GEN_API_KEY = "AIzaSyDMSeg7OQWBtMn6bMjxj3zueZRIdO5udOY"
genai.configure(api_key=GEN_API_KEY)

def explain_url_result(url, result, confidence, model_probs):
    """
    Uses Gemini 2.5 Flash to explain WHY a URL is phishing or legitimate
    """

    prompt = f"""
You are a cybersecurity expert.

Analyze the URL below and explain clearly why it is classified as {result}.

URL:
{url}

Model analysis:
- Structural model phishing probability: {model_probs['m1']:.2f}
- Advanced model phishing probability: {model_probs['m2']:.2f}
- Massive model phishing probability: {model_probs['m3']:.2f}
- Ensemble confidence: {confidence:.2f}

Rules:
- No technical jargon
- 3–5 bullet points
- User-friendly explanation
- Mention visible URL patterns (length, symbols, subdomains, https misuse, etc.)
- max of 3 sentence
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()
