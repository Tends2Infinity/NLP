
################################################################
#                  Word (whitespace) Tokenizer 
################################################################

def word_tokenize(sentence):
    # Splitting the sentence into tokens based on whitespace
    tokens = sentence.split()
    return tokens

sentence = "AI, in 2024, is revolutionizing data analytics."
tokens = word_tokenize(sentence)
print(tokens)

################################################################
#                Character Tokenizer 
################################################################

def character_tokenize(sentence):
    # Decomposing the sentence into individual characters
    tokens = list(sentence)
    return tokens

sentence = "AI, in 2024, is revolutionizing data analytics."
tokens = character_tokenize(sentence)
print(tokens)

################################################################
#                   Punctuation-based Tokenizer 
################################################################

import re

def punctuation_tokenize(sentence):
    # Decomposing the sentence into phrases delimited by punctuation 
    # marks- comma ',' and period '.'
    tokens = re.split(r'[,.]', sentence)
    # Removing empty strings and stripping whitespace from each token
    tokens = [token.strip() for token in tokens if token]
    return tokens

sentence = "AI, in 2024, is revolutionizing data analytics."
tokens = punctuation_tokenize(sentence)
print(tokens)

################################################################
#                    Dictionary-based Tokenizer 
################################################################

def dictionary_tokenize(sentence, dictionary):
    # Splitting the sentence based on whitespaces
    words = sentence.split()
    tokens = []
    unknown_tokens = []
    
    for word in words:
        # Stripping the the words from punctuations ',' & '.'
        stripped_word = word.strip(".,")
        if stripped_word in dictionary:
            tokens.append(stripped_word)
        else:
            unknown_tokens.append(stripped_word)
            
    return tokens, unknown_tokens

# Example dictionary
dictionary = {"AI", "2024", "revolutionize", "data", "analytics"}
sentence = "AI, in 2024, is revolutionizing data analytics."
tokens, unknown_tokens = dictionary_tokenize(sentence, dictionary)

print("Recognized Tokens:", tokens)
print("Unknown Tokens:", unknown_tokens)

################################################################

from collections import defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in v_in:
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = v_in[word]
    return v_out

# Sample vocabulary (word, frequency)
vocab = {'T o k e n i z a t i o n </w>': 1, 'i s </w>': 1, 'c r u c i a l </w>': 1, 'f o r </w>': 1, 'N L P </w>': 1, 't a s k s </w>': 1}
num_merges = 10  # Number of merges

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

# Output the final vocabulary
print(vocab)

################################################################
#                    Rule-based Tokenizer 
################################################################

#import re

def rule_based_tokenize(sentence):
    # Let's use the following regex pattern for our tokenization
    # For alphanumeric characters, regex expresison '\w+'
    # For monetary mounts, regex expression '\$[\d\.]+'
    # and for non-whitespace characters, regex expression '\S+'
    pattern = r'\w+|\$[\d\.]+|\S+'
    tokens = re.findall(pattern, sentence)
    return tokens

sentence = "On average, an AI tool will cost $100.00 in 2024."
tokens = rule_based_tokenize(sentence)
print(tokens)

################################################################
#                    Sentence Tokenizer 
################################################################

import nltk

# Download the tokenizer model 'punkt'
nltk.download('punkt')  

from nltk.tokenize import sent_tokenize
# sent_tokenize() outputs each sentence as a separate token

text = "AI, in 2024, is revolutionizing data analytics. \
        It's transforming industries, one algorithm at a time. What's next? \
        On average, an AI tool will cost $100.00 in 2024."
sentences = sent_tokenize(text)

for sentence in sentences:
    print(sentence)

################################################################
    

import sentencepiece as spm

# Prepare text for training the model
with open('text.txt', 'w') as f:
    f.write("AI, in 2024, is revolutionizing data analysis.")

# Train SentencePiece model
spm.SentencePieceTrainer.Train('--input=text.txt --model_prefix=m --vocab_size=27')

# Load the trained model
sp = spm.SentencePieceProcessor()
sp.Load('m.model')

# Tokenize the sentence
tokens = sp.EncodeAsPieces("AI, in 2024, is revolutionizing data analysis.")
print(tokens)
