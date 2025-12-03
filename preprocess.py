import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from typing import List

# Ensure NLTK resources
def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

ensure_nltk()

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    cleaned = []
    for token, pos in pos_tags:
        if token.isalnum() and token not in STOPWORDS:
            posn = get_wordnet_pos(pos)
            lemma = LEMMATIZER.lemmatize(token, pos=posn)
            if lemma and lemma not in STOPWORDS:
                cleaned.append(lemma)
    return ' '.join(cleaned)


def preprocess_texts(texts: List[str]) -> List[str]:
    return [preprocess_text(t) for t in texts]


def preprocess_query(query: str) -> str:
    return preprocess_text(query)
