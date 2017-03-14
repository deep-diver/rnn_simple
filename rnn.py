from nltk import sent_tokenize
from nltk import word_tokenize
from urllib import request

def get_text_from_web(url):
    return request.urlopen(url).read().decode('utf8')

def get_text_from_file(filename):
    raw = ""
    with open(filename, 'rt') as file:
        raw = file.read()
    return raw

keep_vocab_freq = 8000

# get raw text from the file, 'anna.txt'
raw_text = get_text_from_file('anna.txt')

# tokenize raw_text into sentences
sentences = sent_tokenize(raw_text)

# tokenize each sentences, and stack them into array
tokenized_sentences = [word_tokenize(sent) for sent in sentences]


