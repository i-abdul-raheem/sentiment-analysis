import nltk
nltk.download('all')

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Precompile resources (done once)
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+')
REMOVE_CHARS = str.maketrans('', '', string.punctuation + string.digits)
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def text_processing(txt: str) -> list:
    # Remove URLs
    txt = URL_PATTERN.sub('', txt)
    
    # Remove punctuation + digits
    txt = txt.translate(REMOVE_CHARS)
    
    # Tokenize (lowercase first for speed)
    tokens = word_tokenize(txt.lower())
    
    # Remove stopwords + lemmatize
    tokens = [
        LEMMATIZER.lemmatize(token) 
        for token in tokens if token not in STOP_WORDS
    ]
    
    return tokens
