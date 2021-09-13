# -*- coding: utf-8 -*-
"""
Fonction coût permettant de calculer le seuil sur le score
qui maximise le profit.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


col_list = ['SK_ID_CURR', 'AMT_CREDIT', 'TARGET', 'PREDICTIONS']
df = pd.read_csv('output/oof_model2_04.csv', usecols=col_list)

# Montant moyen accordé
#avg_amount = df['AMT_CREDIT'].mean()

# Total de clients
n = len(df)

# Profits et coûts 
cost = -1       # prêt accordé et non remboursé
profit = 0.1   # prêt accordé et remboursé

results=[]
for profit in [0.15, 0.1, 0.05]:
    for q in range(101):
        thresh = q/100
        col = f'thresh={thresh}'
        df[col] = np.where(df['PREDICTIONS']>=thresh, 1, 0)
        accuracy = accuracy_score(df['TARGET'],df[col])
        #avg_amount=[]
        
        #p = len(df[df[col]==1])/n
        TN = len(df[(df[col]==0) & (df['TARGET']==0)])/n  # True negatives: clients acceptés et qui rapportent du profit
        FN = len(df[(df[col]==0) & (df['TARGET']==1)])/n  # False negatives: clients acceptés et qui engendrent de la perte
        #TP = len(df[(df[col]==1) & (df['TARGET']==1)])/n
        #avg_profit_applicant = (1-TP)*profit + TP*cost
        avg_profit_applicant = profit*TN + cost*FN        # Profit moyen par client lorsque le crédit est accordé
    
        #avg_amt_applicant=[]
        accepted = FN + TN                              # Total des crédits accordés
        avg_profit = avg_profit_applicant * accepted    # Profit réalisé moyen
        results.append([thresh, TN, FN, accepted, accuracy, avg_profit_applicant, avg_profit, profit])
        
        #precision = TP/(TP+FP)
        #recall = TP/(TP+FN)
    
liste_col = ['threshold', 'True negatives %', 'False negatives %', 'Credit accepted %', 'Accuracy', 'Average profit per applicant', 'Average profit total', 'profit']
df_results_per_threshold = pd.DataFrame(results, columns=liste_col)


def lineplot(y, x='threshold'):

    #sns.set(rc={'figure.figsize':(15,10)})
    f = plt.figure(figsize=(10, 7))
    sns.set_style("ticks")
    # plot histogram 
    sns.lineplot(x='threshold', y=y, data=df_results_per_threshold, hue = 'profit', palette='viridis')
    
    for p in df_results_per_threshold['profit'].unique():
        df=df_results_per_threshold[df_results_per_threshold['profit']==p]
        # adding a vertical line for the average passengers per flight
        argmax = df[y].idxmax()
        opt_x = df.loc[argmax, x]
        dy = (df[y].max()+0.04)/(0.04+0.07)*0.95
        plt.axvline(opt_x, ymax=dy, color='#23a62c', label='optimal threshold', linestyle='--')
        
            # adding data label to mean line
        plt.text(x = opt_x*1.03, # x-coordinate position of data label, adjusted to be 3 right of the data point
         #y = max([h.get_height() for h in ax.patches]), # y-coordinate position of data label, to take max height 
         y = 1.02*df[y].max(), # y-coordinate position of data label, to take max height 
         s = 'optimal threshold: {:.2f}'.format(df.loc[argmax, x]), # data label
         color = 'black') # colour of the vertical mean line
    f.show()
    

lineplot("Average profit total")
lineplot("Average profit per applicant")

