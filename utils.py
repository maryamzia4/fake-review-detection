import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def text_process(review):
    nopunc = ''.join([char for char in review if char not in string.punctuation])
    return [word for word in nopunc.split() if word.lower() not in stop_words]
