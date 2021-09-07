# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:02:52 2021

@author: admin
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_table_experiments as dt

from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go

#df = pd.read_csv('output/oof_model2_04.csv')
df = pd.read_csv('output/df_reduced.csv')
feat_importance = pd.read_csv('output/feature_importance_model2_04.csv')
df2=df.iloc[:10,:10]


"""-------Dash app-------"""

app = dash.Dash()

app.layout = html.Div([
    
    html.H1(
        children="Client Loan Management",
        className="header-title",
        style = {
            'textAlign' : 'left'
            }
        ),
    
    # Dropdown sélection client
    dcc.Dropdown(
    id='demo-dropdown',
    options=[
        {'label': i, 'value': i} for i in df['SK_ID_CURR']
        #{'label': 'New York City', 'value': 'NYC'},
        #{'label': 'Montreal', 'value': 'MTL'},
        #{'label': 'San Francisco', 'value': 'SF'}
    ],
    #value='NYC',
    placeholder="Sélectionner un client",
    ),
    
    html.Div(id='dd-output-container'),


    # Graphe
    dcc.Graph(
        id ='samplechart',
        figure = {
            'data' : [
                {'x' : [4,6,8], 'y':[12,16,18], 'type': 'bar', 'name': 'First Chart'},
                {'x' : [2,3,7], 'y':[15,12,14], 'type': 'bar', 'name': 'First Chart'}
                ],
            'layout': {
                'title': 'Simple Chart'
                }
            }
        ),
    
    # Data table
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df2.columns],
        data=df2.to_dict('records'),
        ),
    
    html.Div(id="div-1")
])

@app.callback(
    Output('dd-output-container', 'children'),
    #Output('datatable', 'data'),
    Input('demo-dropdown', 'value')
    )

def update_value(value):
    return 'Vous avez sélectionné le client {}'.format(value)

def update_table(df):
    
    return df.iloc[0].to_dict('records')
#gdef update_df(value):
#    data_client = df[df['SK_ID_CURR']==value]['AMT_CREDIT']
#    return f'Montant du crédit:{data_client}'

if __name__ == "__main__":
    app.run_server(debug=True)

