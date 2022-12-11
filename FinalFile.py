#Importing our libraries that would be used

import datetime
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import figure

#Loading in our dataset
df = pd.read_csv('international_matches_data.csv')

#Converting the column 'date' into type datetime
df['date'] = pd.to_datetime(df['date'])

#Carlos had created a function to fill in any values that have NaN with the mean of that column
def get_mean_and_fill_NaN(df, col):
    mean_value=df[col].mean()
    df[col].fillna(value=mean_value, inplace=True)

#A new dataframe that would be filled with those values was created, however I had decided not to use it
df_w_dummies = pd.read_csv('international_matches_data_w_dummies.csv')
df_w_dummies.drop(columns=df.columns[0], axis=1, inplace=True)

#call get_mean_and_fill_NaN function, filling in those NaN values with the mean using the function Carlos had created
get_mean_and_fill_NaN(df, 'home_team_mean_defense_score')
get_mean_and_fill_NaN(df, 'home_team_goalkeeper_score')
get_mean_and_fill_NaN(df, 'home_team_mean_offense_score') 
get_mean_and_fill_NaN(df, 'away_team_goalkeeper_score') 
get_mean_and_fill_NaN(df, 'home_team_mean_midfield_score') 
get_mean_and_fill_NaN(df, 'away_team_mean_defense_score') 
get_mean_and_fill_NaN(df, 'away_team_mean_offense_score') 
get_mean_and_fill_NaN(df, 'away_team_mean_midfield_score')

#Created a testing dataset that would be used with RandomForest with the values from the original dataset.
Testing_Set = df[['home_team_goalkeeper_score','away_team_goalkeeper_score','home_team_mean_defense_score',
                  'home_team_mean_offense_score','home_team_mean_midfield_score','away_team_mean_defense_score'
                 ,'away_team_mean_offense_score','away_team_mean_midfield_score']]

#Creating two columns that would later help with Random Forest
#Column 'Score Difference' just takes the score difference between the home and away team
#Column 'is_won' would mean that if the home team had a score of > 0, then they had won the match.
#If there was a 0 or a negative number, it would be considered a loss for the home team and a victory for the away team.
df['score_difference'] = df['home_team_score'] - df['away_team_score']
df['is_won'] = df['score_difference'] > 0 

# Load the data
X, y = df.loc[:,['home_team_mean_defense_score','home_team_goalkeeper_score','home_team_mean_offense_score',
                'away_team_goalkeeper_score','home_team_mean_midfield_score','away_team_mean_defense_score',
                'away_team_mean_offense_score','away_team_mean_midfield_score']], df['is_won']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

# Train the random forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

#Printing the accuracy of our model
#In terms of sports, an average of 65% is not terrible as anything above an 80-90% would mean there is some sort of fixing being 
#involved
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

from tqdm import tqdm

#Creating lists to hold winners of each round.
Sim_Winners = list()
Winner_Result = list() 
Semifinal_Winner = list()
Semifinal_Sim = list()

#The number of simulations that we had decided 
n_simulations = 10 

for j in tqdm(range(n_simulations)):
    
    #The final remaining teams in the World Cup as of now. Date : December 12 2022
    candidates = [ 'Croatia', 'Argentina' , 'Morocco' , 'France'] 
    #The final remaining rounds in the World Cup as of now, Date: December 12 2022
    rounds = [ 'semifinal' , 'final']

    for f in rounds:
            iterations = int(len(candidates) / 2)
            #A bracket to hold the winners
            winners = []
            #A bracket to hold the probability 
            prob = []


            for i in range(iterations):
                #Making sure that the home and away team can not be the same team
                #Away will have +1 added to make sure the opponent they play is who they actually play.
                home = candidates[i*2]
                away = candidates[i*2+1]
                #The win prob will then be decided using Random Forest with the Testing_Set that was created way before this
                Win_Probability = model.predict_proba(Testing_Set)[:,1][0]
                Outcome = np.random.binomial(1, Win_Probability)
                winners.append(away) if Outcome <= 0.5 else winners.append(home)
                prob.append(1 - Outcome) if Outcome <= 0.5 else prob.append(Outcome)
                                       
            if f == 'semifinal':    
                Round_Semifinal = ['semifinal'] * 2
                #Taking the semifinal candidates 
                Semifinal_Candidates = zip(Round_Semifinal,winners, prob)
                #Creating a dataframe to show the possible outcome
                Semifinal_Candidates_DF = pd.DataFrame(Semifinal_Candidates, columns = ['Step','Team','Prob'])
                #Adding the results into the Dataframe
                Semifinal_Sim.append(Semifinal_Candidates_DF)
                #Adding the winners of the Semifinals into a list. 
                Semifinal_Winner.append(winners)

            if f == 'final':    
                Round_Final = ['final'] * 1
                #Taking the semifinal candidates 
                Final_Candidates = zip(Round_Final,winners, prob)
                #Creating a dataframe to show the possible outcome
                Final_Candidates_DF = pd.DataFrame(Final_Candidates, columns = ['Step','Team','Prob'])
                #Adding the results into the Dataframe
                Sim_Winners.append(Final_Candidates_DF)
                #Adding the winners of the Semifinals into a list. 
                Winner_Result.append(winners)


            candidates = winners
            

#Combining multiple ranges and strings into the dataframe
Semifinal_Candidates_DF = pd.concat(Semifinal_Sim)
Final_Candidates_DF = pd.concat(Sim_Winners)
Result_DF = pd.concat([Semifinal_Candidates_DF,Final_Candidates_DF]) 

#Adding the number of times team X was predicted as winners for the round
Semifinal_Winner = sum(Semifinal_Winner, [])
Final_Winner = sum(Winner_Result, [])


#Create a list to hold the remaining rounds left of the world cup
lst_results = [Semifinal_Winner,Final_Winner]

#For loop to show which team had won the most times from the simulation
for i in lst_results:
    results = Counter(i).most_common()
    x,y = zip(*results)

    fig,ax = plt.subplots(figsize=(16,8))
    ax.barh(x, y)
    ax.set_ylabel('Team')

    if i == Semifinal_Winner:
        ax.set_title('SemiFinal World Cup 2022')
    if i == Final_Winner:
        ax.set_title('Winner World Cup 2022')

        
    plt.show()