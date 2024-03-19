import os
import math
import nltk
#nltk.download()
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Define the corpus directory
CORPUS_DIR = './US_Inaugural_Addresses'

# Define tokenizer, stopwords and stemmer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

def preprocess_document(filename):
    # Read and preprocess a document from the corpus
    try:
        with open(os.path.join(CORPUS_DIR, filename), "r", encoding='windows-1252') as file:
            doc = file.read().lower()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except IOError:
        print(f"Error reading file: '{filename}'.")
        return []

    tokens = tokenizer.tokenize(doc)
    tokens = [porter_stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens


def getidf(token):
    #Compute the inverse document frequency of a token
    token = porter_stemmer.stem(token)
    count = 0
    N = os.listdir(CORPUS_DIR)
    for filename in N:
        tokens = preprocess_document(filename)
        if token in tokens:
            count += 1
    if count == 0:
        return 0
    else:
        return math.log(len(N) / count) 


def getweight(filename, token):
    #Compute the TF-IDF weight of a token 
    token = porter_stemmer.stem(token)
    tokens = preprocess_document(filename)
    if token not in tokens:
        return 0
    else:
        tf = 1 + math.log(tokens.count(token))  
        idf = getidf(token)
        return tf * idf


def query(qstring):
    #Compute the query vector
    qtokens = tokenizer.tokenize(qstring.lower())
    qtokens = [porter_stemmer.stem(token) for token in qtokens if token not in stop_words]
    scores = {}

    for filename in os.listdir(CORPUS_DIR):
        score = 0
        for qtoken in qtokens:
            w = getweight(filename, qtoken)
            score += w
        scores[filename] = score

    if not scores:
        print("Error: No matching documents found.")
        return None

    max_score = max(scores.values())
    for filename, score in scores.items():
        if score == max_score:
            norm_factor = 1 / math.sqrt(sum([score ** 2 for score in scores.values()]))  # Cosine normalization
            return (filename, norm_factor)

    return None


# Test the functions
print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt', 'arrive'))
print("%.12f" % getweight('07_madison_1813.txt', 'war'))
print("%.12f" % getweight('12_jackson_1833.txt', 'union'))
print("%.12f" % getweight('09_monroe_1821.txt', 'british'))
print("%.12f" % getweight('05_jefferson_1805.txt', 'public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))
