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
avg_amount = df['AMT_CREDIT'].mean()

# Proportion de true positives
p = len(df[df['TARGET']==1])/len(df)

# Profits et coûts 
cost = -1       # prêt accordé et non remboursé
profit = 0.35   # prêt accordé et remboursé

results=[]
for n in range(101):
    thresh = n/100
    col = f'thresh={thresh}'
    df[col] = np.where(df['PREDICTIONS']>=thresh, 1, 0)
    accuracy = accuracy_score(df['TARGET'],df[col])
    #avg_amount=[]
    
    p = len(df[df[col]==1])/len(df)
    TP = len(df[(df[col]==1) & (df['TARGET']==1)])/len(df)
    #avg_profit_applicant = (1-TP)*profit + TP*cost
    avg_profit_applicant = (1-p)*profit + p*cost

    #avg_amt_applicant=[]
    accepted = (1-p)
    avg_profit = avg_profit_applicant * accepted
    results.append([thresh, p, TP, accepted, accuracy, avg_profit_applicant, avg_profit])
    
liste_col = ['threshold', 'positive class %', 'true positive %', 'accepted %', 'accuracy', 'avg_profit_applicant', 'avg_profit']
df_results_per_threshold = pd.DataFrame(results, 
                                        columns=liste_col)
plt.figure()
sns.lineplot(x="threshold", y="accuracy",
             data=df_results_per_threshold)
plt.figure()
sns.lineplot(x="threshold", y="avg_profit_applicant",
             data=df_results_per_threshold)

plt.figure()
sns.lineplot(x="threshold", y="avg_profit",
             data=df_results_per_threshold)

plt.figure()
sns.lineplot(x="threshold", y="positive class %",
             data=df_results_per_threshold)

plt.figure()
sns.lineplot(x="threshold", y="true positive %",
             data=df_results_per_threshold)

df_head=df.head(100)

tp_df = df[(df['thresh=0.02']==1) & (df['TARGET']==1)]

df_test = df[['TARGET', 'PREDICTIONS', 'thresh=0.99']]
