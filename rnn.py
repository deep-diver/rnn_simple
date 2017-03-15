import numpy as np
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

def softmax(x):
    score_mat_exp = np.exp(np.asarray(x))
    return score_mat_exp / score_mat_exp.sum(0)

keep_vocab_freq         = 8000
unknown_token           = "UNKNOWN_TOKEN"
sentence_start_token    = "SENTENCE_START"
sentence_end_token      = "SENTENCE_END"

# get raw text from the file, 'anna.txt'
raw_text = get_text_from_file('anna.txt')

# tokenize raw_text into sentences
sentences = sent_tokenize(raw_text)

# prepend special tokens (start/end of sentences)
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

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

# replace infrequent words with special string, unknown_token
for i, sentence in enumerate(tokenized_sentences):
    for j, word in enumerate(sentence):
        if (word not in word_to_index):
            tokenized_sentences[i][j] = unknown_token

# list[:-1] => exclude the last item
# list[1:] => exclude the first item
x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_label = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# 1 sample training and label
print("x_train[15] x: ")
print(tokenized_sentences[15][:-1])
print(x_train[15])

print("x_train[15] y: ")
print(tokenized_sentences[15][1:])
print(y_label[15])

class RNN:
    def __init__(self, input_dim, hidden_dim=100, bptt_truncate=4):
        self.input_dim      = input_dim
        self.hidden_dim     = hidden_dim
        self.bptt_truncate  = bptt_truncate

        # Random initialization of the weights for each layer
        # numpy.random.uniform(low, high, size)
        self.U = np.random.uniform(-np.sqrt(1./input_dim), 
                                    np.sqrt(1./input_dim), 
                                    (hidden_dim, input_dim))

        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), 
                                    np.sqrt(1./hidden_dim), 
                                    (input_dim, hidden_dim))

        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), 
                                    np.sqrt(1./hidden_dim), 
                                    (hidden_dim, hidden_dim))        

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
    
    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def forward_pass(self, inputs):
        # The total number of time steps 
        # (simple vector size)
        time_steps = len(inputs)

        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        hidden_state = np.zeros((time_steps + 1, self.hidden_dim))
        hidden_state[-1] = np.zeros(self.hidden_dim)

        # The outputs at each time step. Again, we save them for later.
        outputs = np.zeros((time_steps, self.input_dim))

        # For each time step...
        for time_step in np.arange(time_steps):
            # each word has unique index (word_to_index) / 
            # if vocab is "h,e,l,o", then word_to_index['e'] is 1, and one hot encode for 'e' is [0, 1, 0, 0]
            # matrix multiplication between one hot and U is only U part in word_to_index['e'] row
            hidden_from_i = self.U[:,inputs[time_step]]
            hidden_from_h = self.W.dot(hidden_state[time_step-1])   # multiplication between current and previous time state (each memory)           
            hidden_in = hidden_from_i + hidden_from_h               # combine them
            hidden_state[time_step] = np.tanh(hidden_in)            # why tanh? for activation function? => the reason will be updated later.
            
            outputs_in = self.V.dot(hidden_state[time_step])
            outputs[time_step] = softmax(outputs_in)

        return [outputs, hidden_state]

    def predict(self, inputs):
        outputs, hidden_state = self.forward_pass(inputs)
        return np.argmax(outputs, axis=1)

print("forward & predict testing begins")
print("forward_pass(x_train[15])")
np.random.seed(15)
model = RNN(keep_vocab_freq)
o, s = model.forward_pass(x_train[15])
print(o.shape)
print(o)

print("predict(x_train[15])")
predictions = model.predict(x_train[15])
print(predictions.shape)
print(predictions)
print("forward & predict testing ends")