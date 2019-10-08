import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from textblob import TextBlob, Word

def load_dataset(file_name:str, delimiter:str):
    """load dataset from csv file into a pd dataframe"""
    dataset = pd.read_csv(file_name, delimiter)

    return dataset

def dataset_analysis_extension(dataset: pd.DataFrame, save=False):
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
    if save:
        # outputs = dataset['subreddits']
        # del dataset['subreddits']
        # dataset['comments'] = pd.DataFrame(updated_comments)
        # print(dataset)
        tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        dataset = tfidfconverter.fit_transform(updated_comments).toarray()
        # X.to_csv('updated_reddit_train.csv', ',')
    else:
        polarity = []
        # subjectivity = []
        # ngrams = [] # Need to convert to numbers!!!
        # avg_spelling_acc = []
        # word_counts = []
        # for comment in updated_comments:
        #     comment_blob = TextBlob(comment)
        #     polarity.append(comment_blob.sentiment.polarity)
        #     subjectivity.append(comment_blob.sentiment.subjectivity)
        #     # for word in comment_blob.words:
        #     #     print(word.spellcheck())
        #     # print(comment_blob.word_counts)
        #     grams = comment_blob.ngrams(n=6)
        #     ngrams.append(grams)
        # dataset.insert(2, 'polarity', polarity)
        # dataset.insert(2, 'subjectivity', subjectivity)
        # dataset.insert(2, 'ngrams', ngrams)
    return dataset
