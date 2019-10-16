import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import pandas as pd
import datetime
from naive_bayes import NaiveBayes

from dataset import load_dataset, dataset_analysis_extension, clean_dataset, \
    add_pol_sub, pipelinize_feature, named_entry_recognition, get_polarity, \
    get_subjectivity

count_vect_nb = CountVectorizer(ngram_range=(1,2), stop_words="english", binary=True) #nb
print("load dataset")
original_dataset = load_dataset('reddit_train.csv', ',')
test_dataset = load_dataset('reddit_test.csv', ',')

print("splitting x and y")
X_orig = original_dataset.loc[:, original_dataset.columns != 'subreddits']
y = original_dataset['subreddits']
print("clean data")
# X_orig = clean_dataset(X_orig)
with open('clean_train.csv') as f:
    X_orig = [line.strip() for line in f][1:]
with open('clean_test.csv') as f:
    X_test_pipe = [line.strip() for line in f][1:]
ner = load_dataset('ner_clean_train.csv', ',').to_numpy()[:,1:]
ner_test = load_dataset('ner_clean_test.csv', ',').to_numpy()[:,1:]
print("start vectorizer")
X_nb = count_vect_nb.fit_transform(X_orig)

print("start split")
X_train, X_test, y_train, y_test = train_test_split(X_nb, y, test_size=0.01, random_state=42)
nb = NaiveBayes()
print("train naivebayes")
nb.train(X_train, y_train)
print("predict naivebayes")
result = nb.predict(X_test)
print(len(result))
print("accuracy")
accuracy = nb.validation(result, y_test)
print(accuracy)
