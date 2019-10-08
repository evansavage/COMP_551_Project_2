import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from dataset import load_dataset, dataset_analysis_extension

# Load datasets

original_dataset = load_dataset('reddit_train.csv', ',')
# test_dataset = load_dataset('reddit_test.csv', ',')

X = original_dataset.loc[:, original_dataset.columns != 'subreddits']
y = original_dataset['subreddits']

X = pd.DataFrame(dataset_analysis_extension(X, True))
# X_test = pd.DataFrame(dataset_analysis_extension(test_dataset, True))

# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)

# print(updated_dataset)
# print(updated_dataset.shape)
