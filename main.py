import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
import numpy as np
import pandas as pd

from dataset import load_dataset, dataset_analysis_extension, clean_dataset, add_pol_sub

# Load datasets

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

original_dataset = load_dataset('reddit_train.csv', ',')
test_dataset = load_dataset('reddit_test.csv', ',')

X_orig = original_dataset.loc[:, original_dataset.columns != 'subreddits']
y = original_dataset['subreddits']

X_orig = clean_dataset(X_orig)
X_test_pipe = clean_dataset(test_dataset)

X_orig = add_pol_sub(X_orig)

# X = dataset_analysis_extension(X_orig, count_vect, tfidf_transformer)
# X_test = dataset_analysis_extension(X_test_pipe, count_vect, tfidf_transformer, test=True)

print(X.shape, X_test.shape)


# clf = MultinomialNB().fit(X, y)

text_clf = Pipeline([
    ('vect', count_vect),
    ('tfidf', tfidf_transformer),
    ('clf', svm.SVC(gamma='scale')),
])

grid_params = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (5000, 10000, 50000),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    # 'clf__alpha': np.linspace(0.5, 1.5, 6), # For Naive Bayes
    # 'clf__fit_prior': [True, False], # For Naive Bayes
    'clf__decision_function_shape': ('ovo', 'ovr'),
}

gsCV = GridSearchCV(text_clf, grid_params)

# text_clf.fit(X_orig, y)

gsCV.fit(X_orig, y)

# predicted = clf.predict(X_test)
# predicted_pipeline = text_clf.predict(X_test_pipe)

# with open('predictions.csv', 'w') as f:
#     f.write("id,Category\n")
#     for i, item in enumerate(predicted_pipeline):
#         f.write(f"{ i },{ item }\n")


# print(predicted, predicted_pipeline)
print("Best Score: ", gsCV.best_score_)
print("Best Params: ", gsCV.best_params_)

# X_test = pd.DataFrame(dataset_analysis_extension(test_dataset, True))

# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)

# print(updated_dataset)
# print(updated_dataset.shape)
