import pandas as pd 
from sklearn.model_selection import train_test_split


data = pd.read_csv("data.csv")

print(data.head())


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

