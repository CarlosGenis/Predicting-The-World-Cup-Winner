from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('international_matches_data.csv')
#print(df)
#drop first column
df.drop(columns=df.columns[0], axis=1, inplace=True)
#convert dates to 
df['date'] = pd.to_datetime(df['date'], format = '%m/%d/%Y')
#print(df['date'])
#graphs our initial data
#df.hist(bins=50, figsize=(20,15))
#plt.show()

train_set, test_split = train_test_split(df, test_size = 0.2, random_state=42)
#print(train_set)
#print(test_split)
reg = LinearRegression().fit(train_set, test_split)
