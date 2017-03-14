import itertools
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import FreqDist
from urllib import request

def get_text_from_web(url):
    return request.urlopen(url).read().decode('utf8')

def get_text_from_file(filename):
    raw = ""
    with open(filename, 'rt') as file:
        raw = file.read()
    return raw

keep_vocab_freq = 8000
unknown_token = "UNKNOWN_TOKEN"

# get raw text from the file, 'anna.txt'
raw_text = get_text_from_file('anna.txt')

# tokenize raw_text into sentences
sentences = sent_tokenize(raw_text)

# tokenize each sentences, and stack them into array
tokenized_sentences = [word_tokenize(sent) for sent in sentences]

# makes map structures [ word : frequencies ] in descending order
freq_word = FreqDist(itertools.chain(*tokenized_sentences))
print(len(freq_word), "Unique words tokens are Found")

# only keep top keep_vocab_freq number of words
vocab = freq_word.most_common(keep_vocab_freq)

# create index_to_word and word_to_index
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = {c: i for i, c in enumerate(index_to_word)}
