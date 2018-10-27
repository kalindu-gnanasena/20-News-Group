import os
import pandas as pd
from string import punctuation

# libraries importing from scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# natural language toolkit packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.stem.snowball import  EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# get the set of stop words
_stop_words = set(stopwords.words('english') + list(punctuation))

# initialize treebank word tokenizer and detokenizer
tokenizer = TreebankWordTokenizer() 
detokenizer = TreebankWordDetokenizer()

# initialize english stemmer
stemmer = EnglishStemmer()

# initialize word net lemmatizer
lemmatizer = WordNetLemmatizer()

# word list used in english language
_word_list = set([word for word in wn.words(lang='eng')])

# method to get all the text files from the directory to a list
def read_files(dir):
    f = []
    for roots, dirs, files in os.walk(dir):
        for file in files:
            f.append(os.path.join(roots, file))
    return f

# method to categorize and read the files
def open_files(file_paths):
    comp = []
    rec = []
    sci = []
    talk = []
    other = []
    
    for path in file_paths:
        if "comp" in path:
            w = open(path, encoding='utf-8', errors='ignore')
            comp += [w.read()]
        elif "rec" in path:
            w = open(path, encoding='utf-8', errors='ignore')
            rec += [w.read()]
        elif "sci" in path:
            w = open(path, encoding='utf-8', errors='ignore')
            sci += [w.read()]
        elif "talk" in path:
            w = open(path, encoding='utf-8', errors='ignore')
            talk += [w.read()]
        elif "alt" in path or "misc" in path or "soc" in path:
            w = open(path, encoding='utf-8', errors='ignore')
            other += [w.read()]
    return comp, rec, sci, talk, other

# method to clean data
def clean_data(data):
    cleaned_data = []
    for text in data:
        words = []
        for word in tokenizer.tokenize(text):
            word = stemmer.stem(word)
            word = lemmatizer.lemmatize(word)
            words += [word]   
        cleaned_data += [detokenizer.detokenize([word for word in words if word in _word_list if word.isalpha()])]
    return cleaned_data

# frequency vectorizer
def count_vectorize_csv(cleaned_data, name):
    vectorizer = CountVectorizer(stop_words=_stop_words)
    x = vectorizer.fit_transform(cleaned_data)
    y = vectorizer.get_feature_names()
    
    df = pd.DataFrame(data=x.toarray(), columns=y)
    
    train, test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42)
    
    train.to_csv(name+'_count_train.csv')
    test.to_csv(name+'_count_test.csv')
    
    print('Count vectorize train and test csv for {0} is successfully created!'.format(name))

# binary vectorizer
def binary_vectorize_csv(cleaned_data, name):
    vectorizer = CountVectorizer(stop_words=_stop_words, binary=True)
    x = vectorizer.fit_transform(cleaned_data)
    y = vectorizer.get_feature_names()
    
    df = pd.DataFrame(data=x.toarray(), columns=y)
    
    train, test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42)
    
    train.to_csv(name+'_binary_train.csv')
    test.to_csv(name+'_binary_test.csv')
    
    print('Binary vectorize train and test csv for {0} is successfully created!'.format(name))

# tfidf vectorizer
def tfidf_vectorize_csv(cleaned_data, name):
    vectorizer = TfidfVectorizer(stop_words=_stop_words)
    x = vectorizer.fit_transform(cleaned_data)
    y = vectorizer.get_feature_names()
    
    df = pd.DataFrame(data=x.toarray(), columns=y)
    
    train, test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42)
    
    train.to_csv(name+'_tfidf_train.csv')
    test.to_csv(name+'_tfidf_test.csv')
    
    print('Tfidf vectorize train and test csv for {0} is successfully created!'.format(name))

# calling frequency, binary and tfidf vectorizers
def get_feature_vectors(data, name):
    cleaned_data = clean_data(data)
    
    count_vectorize_csv(cleaned_data, name)
    binary_vectorize_csv(cleaned_data, name)
    tfidf_vectorize_csv(cleaned_data, name)


# Running the programme
def run_main(file_path):

    files = read_files(file_path)

    # define lists to store data
    comp_data = []
    rec_data = []
    sci_data = []
    talk_data = []
    other_data = []

    # open files
    comp_data, rec_data, sci_data, talk_data, other_data = open_files(files)

    get_feature_vectors(comp_data, 'comp')
    get_feature_vectors(rec_data, 'rec')
    get_feature_vectors(sci_data, 'sci')
    get_feature_vectors(talk_data, 'talk')
    get_feature_vectors(other_data, 'other')


run_main('./')