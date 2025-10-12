import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def evaluate_prompt(prompt):
    words = prompt.split()
    stopword_ratio = sum(1 for w in words if w in stopwords.words('english')) / len(words)
    clarity = round((1 - stopword_ratio) * 10, 2)
    return {"clarity_score": clarity}
