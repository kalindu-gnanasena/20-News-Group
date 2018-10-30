# 20 News Group - Feature Extraction

## Process
```
1. Download the dataset from given URL below <br>
2. Extract it.
3. Identify the data distribution.
4. Categorized data as required.
5. Import relevant packages.
6. Read data from files and get it to a list. (Use relevant URL encoding patterns to avoid buffer exceptions.)
7. Remove Stop words and other irrelevant parts using NLTK functions.
8. Tokenize Words.
9. Vectorize and divide Train and Test sets.
10. Write data to CSV using pandas data frames.
```

## Overview:

**This dataset is a collection of  20 newsgroups documents. The processing has been done for the purpose of feature extraction.**

*List of the 20 newsgroups:*
```
- comp.graphics
- comp.os.ms-windows.misc
- comp.sys.ibm.pc.hardware
- comp.sys.mac.hardware
- comp.windows.x rec.autos
- rec.motorcycles
- rec.sport.baseball
- rec.sport.hockey sci.crypt
- sci.electronics
- sci.med
- sci.space
- misc.forsale talk.politics.misc
- talk.politics.guns
- talk.politics.mideast talk.religion.misc
- alt.atheism
- soc.religion.christian
```
#### Download Link: [20news-bydate.tar.gz ](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) - 20 Newsgroups sorted by date; duplicates and some headers removed (18846 documents)

The 20 newsgroup dataset was transformed by using the **Bag of word** and **Term frequency-Inverse document frequency (tf-idf)** method. The dataset after transformation consists of five main classes:
 ```   
 - Computer
 - Recreational
 - Science
 - Talk show
 ``` 
 and each of these classes contains **`train.csv`** and **`test.csv`** files.
 
## Libraries used

 - `os` - directory access
 - `pandas` - writing to csv
 - `nltk` - processing toolkit for NLP tasks
 - `sklearn` - vectorize and split train, test sets


## Modifications
 
 **Used Functions**
 
 - [x] Bag of words - `CountVectorizer()`
 - [x] Bag of Words (Binary) - `CountVectorizer(binary=True)`
 - [x] TF-IDF - `TfidfVectorizer()`


#### Generated CSV Files:

`comp_count_train.csv` `comp_count_test.csv` `comp_binary_train.csv` `comp_binary_test.csv` `comp_tfidf_train.csv` `comp_tfidf_test.csv`

`rec_count_train.csv` `rec_count_test.csv` `rec_binary_train.csv` `rec_binary_test.csv` `rec_tfidf_train.csv` `rec_tfidf_test.csv`

`sci_count_train.csv` `sci_count_test.csv` `sci_binary_train.csv` `sci_binary_test.csv` `sci_tfidf_train.csv` `sci_tfidf_test.csv`

`talk_count_train.csv` `talk_count_test.csv` `talk_binary_train.csv` `talk_binary_test.csv` `talk_tfidf_train.csv` `talk_tfidf_test.csv`
