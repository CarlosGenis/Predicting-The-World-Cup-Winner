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

#this code is forthe purpose of one-hot encoding
#test = pd.get_dummies(df, drop_first=True)
#print(test)
#test.to_csv('international_matches_data_w_dummies.csv')

#function to get mean of a column and fills in Nan values

def get_mean_and_fill_NaN(df, col):
    mean_value=df[col].mean()
    df[col].fillna(value=mean_value, inplace=True)


#using our new df
df_w_dummies = pd.read_csv('international_matches_data_w_dummies.csv')
df_w_dummies.drop(columns=df.columns[0], axis=1, inplace=True)

#call get_mean_and_fill_NaN function
get_mean_and_fill_NaN(df_w_dummies, 'home_team_goalkeeper_score') 
test = df_w_dummies['home_team_goalkeeper_score']
#train_set, test_split = train_test_split(df_w_dummies, test_size = 0.2, random_state=42)
#test = train_set
#test2 = test_split

