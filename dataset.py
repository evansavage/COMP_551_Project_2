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
    lemmatizer = WordNetLemmatizer()
    for comment in comments:
        comment = re.sub(r'\d+', '', comment)
        comment = re.sub(r'\s+[a-zA-Z]\s+', ' ', comment)
        comment = re.sub(r'\^[a-zA-Z]\s+', ' ', comment)
        comment = re.sub(r'\s+', ' ', comment, flags=re.I)
        comment = comment.lower()
        comment = comment.translate(translator)
        comment = comment.strip()
        comment_blob = TextBlob(comment)
        comment_blob = [i for i in comment_blob.words if not i in stop_words]
        final_string =''
        for word in comment_blob:
            final_string += lemmatizer.lemmatize(word) + ' '
        updated_comments.append(final_string)
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
    return ne_chunk(pos_tag(word_tokenize(input_str)))

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
    return TextBlob(text).sentiment.polarity

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def reshape_a_feature_column(series):
    return np.reshape(np.asarray(series), (len(series), 1))

def pipelinize_feature(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            processed = [function(i) for i in list_or_series]
            processed = reshape_a_feature_column(processed)
        else: # if it's not active, just pass it right back
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})
