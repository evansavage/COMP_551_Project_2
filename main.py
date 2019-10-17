import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
import numpy as np
import pandas as pd
import datetime
import csv

from dataset import load_dataset, dataset_analysis_extension, clean_dataset, \
    add_pol_sub, pipelinize_feature, named_entity_recognition, get_polarity, \
    get_subjectivity, ner_input, get_comment_length

print('Use existing clean dataset? [y/n]')
clean = input()
print('Is this run for tuning or for predicting? [t/p]')
tuning = input()
# Load datasets

count_vect = CountVectorizer(stop_words="english")
tfidf_transformer = TfidfTransformer(use_idf=True, norm='l2')
tfidf_vec = TfidfVectorizer(stop_words="english")

original_dataset = load_dataset('reddit_train.csv', ',')
test_dataset = load_dataset('reddit_test.csv', ',')

X = original_dataset.loc[:, original_dataset.columns != 'subreddits']
y = original_dataset['subreddits']

if clean == 'n':
    X_orig = clean_dataset(X)
    X_test_pipe = clean_dataset(test_dataset)
    ner = named_entity_recognition(X_orig)
    ner_test = named_entity_recognition(X_test_pipe)

    with open('clean_train.csv', 'w') as f:
        f.write("train\n")
        for entry in X_orig:
            f.write(f"{ entry }\n")

    with open('clean_test.csv', 'w') as f:
        f.write("test\n")
        for entry in X_test_pipe:
            f.write(f"{ entry }\n")

    ner.to_csv('ner_clean_train.csv')
    ner_test.to_csv('ner_clean_test.csv')

elif clean == 'y':
    with open('clean_train.csv') as f:
        X_orig = [line.strip() for line in f][1:]
    with open('clean_test.csv') as f:
        X_test_pipe = [line.strip() for line in f][1:]
    ner = load_dataset('ner_clean_train.csv', ',').to_numpy()[:,1:]
    ner_test = load_dataset('ner_clean_test.csv', ',').to_numpy()[:,1:]

ner = ner / ner.max(axis=0)
ner_test = ner_test / ner_test.max(axis=0)
print(ner)

lda = LatentDirichletAllocation(n_components=20, random_state=0)
rfe = RFE(MultinomialNB())
# lda.fit_transform(np.array(X_orig).reshape(-1,1), y)

text_clf = Pipeline([
    ('features', FeatureUnion([
        # ('reg', Pipeline([
        #     ('vect', count_vect),
        #     ('tfidf', tfidf_transformer),
        # ])),
        ('reg', Pipeline([
            ('tfvec', tfidf_vec)
        ]))
        # ('topics', Pipeline([
        #     ('vect', count_vect),
        #     ('top', lda),
        # ])),
        # ('ner', ner_input(ner, ner_test, active=True)),
        # ('len', pipelinize_feature(get_comment_length, active=True)),
        # ('polarity', pipelinize_feature(get_polarity, active=True)),
        # ('subjectivity', pipelinize_feature(get_subjectivity, active=True)),
    ])),
    # ('rfe', rfe),
    ('clf', svm.LinearSVC()),
])

grid_params = {
    # 'features__reg__vect__max_df': (0.1,0.9, 1),
    # 'features__reg__tfvec__min_df': (2,3),
    # 'features__reg__vect__ngram_range': ((1,2),(1,3)),
    'features__reg__tfvec__max_df': (0.8, 0.9, 1),
    # 'features__reg__tfvec__ngram_range': ((1,2),(1,3)),
    #'features__reg__tfvec__sublinear_tf': (True, False),
    #'features__reg__tfvec__norm': ('l1', 'l2'),
    # 'features__reg__tfvec__max_features': (800000, 1200000),
    # 'features__reg__tfvec__use_idf': (True, False),
    #'features__reg__tfvec__smooth_idf': (True, False)
    # 'features__topics__vect__max_features': (1500, 3000),
    # 'features__topics__vect__max_df': (0.8, 0.9, 1),
    # 'features__topics__vect__min_df': (5,10,12),
    # 'features__topics__vect__ngram_range': ((1,2),(1,3)),
    # 'features__reg__tfidf__use_idf': (True, False),
    # 'features__reg__tfidf__norm': ('l1', 'l2'),
    # 'clf__alpha': np.linspace(1, 1.5, 6), # For Naive Bayes
    # 'clf__fit_prior': [True, False], # For Naive Bayes
    # 'clf__decision_function_shape': ('ovo', 'ovr'), # For svm.SVC
    # 'clf__max_iter': (1100, 5000),
    # 'clf__intercept_scaling': (1.5, 1.8),
    # 'clf__multi_class': ('ovr', 'crammer_singer'),
    # 'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.2, 0.9),
    'clf__dual': (False, True)
    # 'clf__solver': ('newton-cg', 'lbfgs'),
    # 'clf__n_estimators': (10, 100),
    # 'clf__random_state': (0,1),
    # 'clf__max_features': ('auto', 30, 60),
}

if tuning == "t":
    gsCV = GridSearchCV(text_clf, grid_params, cv=5, verbose=3)

    # text_clf.fit(X_orig, y)

    gsCV.fit(X_orig, y)
    print("Best Score: ", gsCV.best_score_)
    print("Best Params: ", gsCV.best_params_)
    # predicted = gsCV.predict(X_test)
    # predicted_pipeline = text_clf.predict(X_test_pipe)

elif tuning == 'p':
    text_clf.set_params(
        features__reg__tfvec__max_df=0.8,
        # features__reg__vect__min_df=5,
        #features__reg__tfvec__ngram_range=(1, 2),
        #clf__max_iter=1100,
        #clf__intercept_scaling=1.5,
        clf__C=0.2,
        clf__dual=False
        # clf__alpha=1.0,
        # clf__fit_prior=True,
        
    )
    text_clf.fit(X_orig, y)
    predicted = text_clf.predict(X_test_pipe)
    with open('predictions.csv', 'w') as f:
        f.write("id,Category\n")
        for i, item in enumerate(predicted):
            f.write(f"{ i },{ item }\n")

    # print(predicted, predicted_pipeline)


    # with open('parameters_history.txt', 'a+') as f:
    #     f.write("Tuning on: ", datetime.datetime.now(), '\n')
    #     f.write(str(grid_params) + '\n')
    #     f.write("Best Score: ", gsCV.best_score_, '\n')
    #     f.write("Best Params: ", gsCV.best_params_, '\n\n\n')
# X_test = pd.DataFrame(dataset_analysis_extension(test_dataset, True))

# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)

# print(updated_dataset)
# print(updated_dataset.shape)
