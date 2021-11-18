import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data = pd.read_csv('data/data.csv')
    train, test = train_test_split(data, test_size=.25)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
