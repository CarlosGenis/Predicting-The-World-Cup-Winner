import pandas as pd

df = pd.read_csv ('results.csv')
#The data set below is for the purposes of getting data for countries that haven't been to the World Cup recentely
df_filtered_comp = df[(df['tournament'] == 'UEFA Euros') | (df['tournament'] == 'AFC Asian Cup') | (df['tournament'] == 'Gold Cup')]
print(df_filtered_comp)