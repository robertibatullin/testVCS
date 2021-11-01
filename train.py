import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    X_train = train[['x','y']]
    y_train = train['label']
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(clf, f)
