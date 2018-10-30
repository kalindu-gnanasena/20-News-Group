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
    return comp, rec, sci, talk


def get_unique_word_set(data_list):
    word_set = []
    for docs in data_list:
        for doc in docs:
            words = []
            for word in word_tokenize(doc):
                word = stemmer.stem(word)
                word = lemmatizer.lemmatize(word)
                words += [word]
            word_set += [word for word in words if word not in _stop_words if word in _word_list if word.isalpha()]
    return list(set(word_set))


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


def count_vectorize_csv(cleaned_data, name, vocabulary_set):
    vectorizer = CountVectorizer(stop_words=_stop_words, vocabulary=vocabulary_set)
    x = vectorizer.fit_transform(cleaned_data)
    y = vectorizer.get_feature_names()
    
    df = pd.DataFrame(data=x.toarray(), columns=y)
    
    df.to_csv(name+'_freq.csv')
    
    print(name+'_freq.csv successfully created!')


def binary_vectorize_csv(cleaned_data, name, vocabulary_set):
    vectorizer = CountVectorizer(stop_words=_stop_words, binary=True, vocabulary=vocabulary_set)
    x = vectorizer.fit_transform(cleaned_data)
    y = vectorizer.get_feature_names()
    
    df = pd.DataFrame(data=x.toarray(), columns=y)
    
    df.to_csv(name+'_binary.csv')
    
    print(name+'_binary.csv successfully created!')


def tfidf_vectorize_csv(cleaned_data, name, vocabulary_set):
    vectorizer = TfidfVectorizer(stop_words=_stop_words, vocabulary=vocabulary_set)
    x = vectorizer.fit_transform(cleaned_data)
    y = vectorizer.get_feature_names()
    
    df = pd.DataFrame(data=x.toarray(), columns=y)
    
    df.to_csv(name+'_tfidf.csv')
    
    print(name+'_tfidf.csv successfully created!')


def get_feature_vectors(data, name, vocabulary_set):
    cleaned_data = clean_data(data)
    
    count_vectorize_csv(cleaned_data, name, vocabulary_set)
    binary_vectorize_csv(cleaned_data, name, vocabulary_set)
    tfidf_vectorize_csv(cleaned_data, name, vocabulary_set)


def run_main(file_path):
    files = read_files(file_path)

    comp_data = []
    rec_data = []
    sci_data = []
    talk_data = []

    comp_data, rec_data, sci_data, talk_data = open_files(files)


    comp_data_train, comp_data_test = train_test_split(comp_data, train_size=0.7, test_size=0.3, random_state=42)
    rec_data_train, rec_data_test = train_test_split(rec_data, train_size=0.7, test_size=0.3, random_state=42)
    sci_data_train, sci_data_test = train_test_split(sci_data, train_size=0.7, test_size=0.3, random_state=42)
    talk_data_train, talk_data_test = train_test_split(talk_data, train_size=0.7, test_size=0.3, random_state=42)


    train_data = [comp_data_train, rec_data_train, sci_data_train, talk_data_train]
    test_data = [comp_data_test, rec_data_test, sci_data_test, talk_data_test]


    unique_word_set = get_unique_word_set(train_data)
    unique_word_set.sort()


    # vectorize comp train and test data
    get_feature_vectors(comp_data_train, 'comp_train', unique_word_set)
    get_feature_vectors(comp_data_test, 'comp_test', unique_word_set)

    # vectorize rec train and test data
    get_feature_vectors(rec_data_train, 'rec_train', unique_word_set)
    get_feature_vectors(rec_data_test, 'rec_test', unique_word_set)


    # vectorize sci train and test data
    get_feature_vectors(sci_data_train, 'sci_train', unique_word_set)
    get_feature_vectors(sci_data_test, 'sci_test', unique_word_set)


    # vectorize talk train and test data
    get_feature_vectors(talk_data_train, 'talk_train', unique_word_set)
    get_feature_vectors(talk_data_test, 'talk_test', unique_word_set)




# execute programme
run_main('./')