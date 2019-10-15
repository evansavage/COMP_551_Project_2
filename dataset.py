import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from textblob import TextBlob, Word

import spacy
from spacy import displacy
from collections import Counter
# import en_core_web_sm

# Downlad: python3 -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
from sklearn.preprocessing import FunctionTransformer

nltk.download('maxent_ne_chunker')
nltk.download('words')

def load_dataset(file_name:str, delimiter:str):
    """load dataset from csv file into a pd dataframe"""
    dataset = pd.read_csv(file_name, delimiter)

    return dataset

def clean_dataset(dataset: pd.DataFrame):
    comments = dataset['comments']
    updated_comments = []
    translator = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    # lemmatizer = WordNetLemmatizer()
    # unique_tags = []
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    for i, comment in enumerate(comments):
        # Remove urls
        comment = re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '', comment)
        # Remove usernames
        comment = re.sub(r'/?u/[A-Za-z0-9_-]+', '', comment)
        # Remove subreddit prefix
        comment = re.sub('/?r/', '', comment)
        # Remove numbers
        comment = re.sub(r'\d+', '', comment)
        # remove single characters
        comment = re.sub(r'\s+[a-zA-Z]\s+', ' ', comment)
        # remove single characters from the start
        comment = re.sub(r'\^[a-zA-Z]\s+', ' ', comment)
        # replace multiple space with single space
        comment = re.sub(r'\s+', ' ', comment, flags=re.I)
        comment = comment.lower()
        # Remove punctuation
        comment = comment.translate(translator)
        comment = comment.strip()
        comment_blob = TextBlob(comment)
        # tags = comment_blob.pos_tags
        # comment_blob = [i for i in comment_blob.words if not i in stop_words]
        # updated = TextBlob(' '.join(comment_blob))
        words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in comment_blob.tags]
        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        # for i,word in enumerate(comment_blob):
        #     # print(tags[i][1])
        #     final_string += word.lemmatize() + ' '
        # # print(final_string)
        # print(i, comment)
        updated_comments.append(' '.join(lemmatized_list))

    print('Done cleaning')
    return updated_comments

def add_pol_sub(clean_dataset:list):
    polarity = []
    subjectivity = []

    for comment in clean_dataset:
        comment_blob = TextBlob(comment)
        polarity.append(comment_blob.sentiment.polarity)
        subjectivity.append(comment_blob.sentiment.subjectivity)

    sPolarity = sp.csr_matrix([x + 1.0 for x in polarity]).transpose() # Naive bayes requires positive numbers
    sSubjectivity = sp.csr_matrix(subjectivity).transpose()

    dataset = sp.hstack((dataset, sPolarity))
    dataset = sp.hstack((dataset, sSubjectivity))

    return dataset

def named_entry_recognition(input_str:str):
    out = nlp(input_str)
    print([(X.text, X.label_) for X in out.ents])
    # return ne_chunk(pos_tag(word_tokenize(input_str)))

def dataset_analysis_extension(clean_dataset:list, countVec, transformer, test=False):

    if test == False:
        dataset_counts = countVec.fit_transform(clean_dataset)
        dataset = transformer.fit_transform(dataset_counts)
    else:
        dataset_counts = countVec.transform(clean_dataset)
        dataset = transformer.transform(dataset_counts)


    print(type(dataset))
        # X.to_csv('updated_reddit_train.csv', ',')
    # else:
    polarity = []
    subjectivity = []
    ngrams = [] # Need to convert to numbers!!!
    avg_spelling_acc = []
    word_counts = []
    for comment in clean_dataset:
        comment_blob = TextBlob(comment)
        polarity.append(comment_blob.sentiment.polarity)
        subjectivity.append(comment_blob.sentiment.subjectivity)

    sPolarity = sp.csr_matrix([x + 1.0 for x in polarity]).transpose() # Naive bayes requires positive numbers
    sSubjectivity = sp.csr_matrix(subjectivity).transpose()

    dataset = sp.hstack((dataset, sPolarity))
    dataset = sp.hstack((dataset, sSubjectivity))

    return dataset

def get_polarity(text):
    return TextBlob(text).sentiment.polarity + 1.0

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def get_comment_length(text):
    return len(text)

def reshape_a_feature_column(series):
    return np.reshape(np.asarray(series), (len(series), 1))


def named_entity_recognition(clean_dataset:list):
    columns = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
        'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
        'QUANTITY', 'ORDINAL', 'CARDINAL']
    ne_dataset = pd.DataFrame(0, index=np.arange(len(clean_dataset)),
        columns=columns)
    for i, entry in enumerate(clean_dataset):
        out = nlp(entry)
        if out.ents:
            for X in out.ents:
                ne_dataset.at[i, X.label_] += 1
        # print(ne_dataset.loc[[i]])
    print(ne_dataset)
    return ne_dataset

def ner_input(train, test, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            if(len(train) > 30000):
                processed = train
            else:
                processed = test
        else:
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})

def pipelinize_feature(function, active=True, matrix=False):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            if not matrix:
                processed = [function(i) for i in list_or_series]
                processed = reshape_a_feature_column(processed)
            else:
                processed = function(list_or_series).to_numpy()
        else: # if it's not active, just pass it right back
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})
