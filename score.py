import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    with open('model/model.pkl', 'rb') as f:
        clf = pickle.load(f)
    test = pd.read_csv('data/test.csv')
    X_test = test[['x','y']]
    y_test = test['label']
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    print(roc_auc_score(y_test, y_pred_proba))
