import sys

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    root = sys.argv[1]
    test_size = float(sys.argv[2])
    data = pd.read_csv(root+'/data.csv')
    train, test = train_test_split(data, test_size = test_size)
    train.to_csv(root+'/train.csv', index=False)
    test.to_csv(root+'/test.csv', index=False)
