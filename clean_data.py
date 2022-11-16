import pandas as pd

df = pd.read_csv ('MatchesData2.csv')
#The data set below is for the purposes of getting data for countries that haven't been to the World Cup recentely
df_wc = df[(df['tournament'] == 'FIFA World Cup')]
#print(df_filtered_comp)
#df_filtered_comp.to_csv("match_data.csv", encoding='utf-8', index=False)
df_canada = df[((df['home_team'] == 'Canada' ) | (df['away_team'] == 'Canada')) & (df['tournament'] == 'Gold Cup')]
df_wales = df[((df['home_team'] == 'Wales' ) | (df['away_team'] == 'Wales')) & (df['tournament'] == 'UEFA Euro')]
df_qatar = df[((df['home_team'] == 'Qatar' ) | (df['away_team'] == 'Qatar')) & (df['tournament'] == 'AFC Asian Cup')]
df_wc_ = df_wc.append(df_canada, ignore_index=True)
df_wc__ = df_wc_.append(df_wales, ignore_index=True)
df_wc___ = df_wc__.append(df_qatar, ignore_index=True)
#print(df_wc___)

#df_wc___.to_csv('match_data.csv')