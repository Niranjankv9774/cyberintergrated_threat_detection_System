# feature_extract.py
from urllib.parse import urlparse

def extract_features_v1(url):
    """Features for Model 1 (Structural - 19 features)"""
    return {
        'url_length': len(url),
        'n_dots': url.count('.'),
        'n_hypens': url.count('-'),
        'n_underline': url.count('_'),
        'n_slash': url.count('/'),
        'n_questionmark': url.count('?'),
        'n_equal': url.count('='),
        'n_at': url.count('@'),
        'n_and': url.count('&'),
        'n_exclamation': url.count('!'),
        'n_space': url.count(' ') + url.count('%20'),
        'n_tilde': url.count('~'),
        'n_comma': url.count(','),
        'n_plus': url.count('+'),
        'n_asterisk': url.count('*'),
        'n_hastag': url.count('#'),
        'n_dollar': url.count('$'),
        'n_percent': url.count('%'),
        'n_redirection': 0  # Optional: can compute redirects if needed
    }

def extract_features_v2_v3(url):
    """Features for Model 2 & 3 (Advanced - 31 features)"""
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path
    
    def count_digits(s):
        return sum(c.isdigit() for c in s)

    features = {
        'length_url': len(url),
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_at': url.count('@'),
        'nb_qm': url.count('?'),
        'nb_and': url.count('&'),
        'nb_or': url.count('|'),
        'nb_eq': url.count('='),
        'nb_underscore': url.count('_'),
        'nb_tilde': url.count('~'),
        'nb_percent': url.count('%'),
        'nb_slash': url.count('/'),
        'nb_star': url.count('*'),
        'nb_colon': url.count(':'),
        'nb_comma': url.count(','),
        'nb_semicolumn': url.count(';'),
        'nb_dollar': url.count('$'),
        'nb_space': url.count(' ') + url.count('%20'),
        'nb_www': 1 if 'www' in url.lower() else 0,
        'nb_com': url.lower().count('.com'),
        'nb_dslash': url.count('//'),
        'http_in_path': 1 if 'http' in path.lower() else 0,
        'https_token': 1 if 'https' in url.lower() else 0,
        'ratio_digits_url': count_digits(url) / len(url) if len(url) > 0 else 0,
        'ratio_digits_host': count_digits(host) / len(host) if len(host) > 0 else 0,
        'punycode': 1 if 'xn--' in url.lower() else 0,
        'port': 1 if parsed.port else 0,
        'nb_subdomains': host.count('.') if host else 0,
        'prefix_suffix': 1 if '-' in host else 0,
        'random_domain': 0,  # Optional: detect random domains
        'shortening_service': 1 if any(s in host for s in ['bit.ly', 'goo.gl', 't.co', 'tinyurl']) else 0
    }
    return features
