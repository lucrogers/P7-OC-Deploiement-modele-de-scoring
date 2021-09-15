# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:02:52 2021

@author: admin
"""

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html


import dash_table
import dash_table_experiments as dt
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go

"""
#df = pd.read_csv('output/oof_model2_04.csv')
feat_importance = pd.read_csv('output/feature_importance_model2_04.csv')
descriptions = pd.read_csv('input/HomeCredit_columns_description.csv')

"""
df = pd.read_csv('output/df_reduced.csv')
infos_clients = ['SK_ID_CURR', 'TARGET','NAME_CONTRACT_TYPE', 'CODE_GENDER', 'AMT_CREDIT', 'OCCUPATION_TYPE']



"""-------Dash parameters-------"""
left_col_width = 4
right_col_width = "100%"

"""-------Dash app-------"""

app = dash.Dash(
    external_stylesheets=[dbc.themes.SLATE]
    #external_stylesheets=[dbc.themes.GRID]
    )

app.layout = dbc.Container([
    
    
    #1 row
    dbc.Row([
        #Title
        dbc.Col([
            #Title
            html.H1("Client Loan Management",
                        className='text-left, mb-4')
            ],) # end col
        ]), # end first row
    
    
    #2 row
    dbc.Row([
        dbc.Col([   
            
            #Dropdown
            #dbc.Card("Dropdown", body=True)  

# =============================================================================
#             dcc.Dropdown(id='user-id-dd',                         
#                          options=[{'label': i, 'value': i} for i in df['SK_ID_CURR']],
#                          placeholder="Sélectionnez un client"
#                                      )
# =============================================================================
  
            dcc.Dropdown(id='user-id-dd',                         
                         options=[{'label': i, 'value': i} for i in df['SK_ID_CURR']],
                         placeholder="Sélectionnez un client"
                                     )

                    
            ],width=left_col_width ), # end first col
        dbc.Col([
            
            #Tabs client/groupe
            dbc.Card("Tabs client/groupe", body=True)  
                      
            ], ) # end 2nd col

        ]), # end 2nd row
    
    html.Br(),

    #3 row
    dbc.Row([
        dbc.Col([
            
            #Datatable infos clients
            #dbc.Card("Datatable infos clients", body=True)
            #dbc.Table.from_dataframe(df0, id='user-table', striped=True, bordered=True, hover=True)
            
# =============================================================================
#             html.Div(id='table',
#                      children=[dash_table.DataTable(
#                     id='user-table',
#                     columns=[{"name": i, "id": i} for i in ['index', '2']],
#                     data=[]
#                          )])
# =============================================================================
            
            html.Div(id='table')
            
                   
           
           ], width=left_col_width), #end 1st col
        
        dbc.Col([
            
            # Graphs
            dbc.Card("Graphs", body=True)  
                      
            ]) # end 2nd col
        
        ]), # end 3rd row
    
    html.Br(),

    #4 row
    dbc.Row([
        dbc.Col([
            
            #Annotations
            dbc.Card("Annotations", body=True) 
                       
            ], width=left_col_width), #end 1st col
        
        dbc.Col([
            
            # Text explications
            dbc.Card("Text explications", body=True)  
                      
            ]) #end 2nd col
        
        ]) # end 4th row       

    
    ]) # end layout container

@app.callback(
    Output('table', 'children'),
    Input('user-id-dd', 'value')
    )
def update_value(user_id):
    if user_id != None:
        user = df[infos_clients][df['SK_ID_CURR']==user_id].transpose().reset_index()
        user.columns = ['Infos', 'Client']
    else:
        user = pd.DataFrame(columns=['Infos', 'Client'])
    #return dbc.Table.from_dataframe(user)
    return dbc.Table.from_dataframe(user, striped=True, bordered=True, hover=True)
   
# =============================================================================
# def update_value(user_id):
#     user = df0[df0['First Name']==user_id]
#     #return dbc.Table.from_dataframe(user)
#     return dbc.Table.from_dataframe(user, striped=True, bordered=True, hover=True)
# =============================================================================

"""
def update_value(user_id):
    user = df[infos_clients][df['SK_ID_CURR']==user_id].transpose().reset_index()
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in user.columns]) ] +
        # Body
        [html.Tr([
            html.Td(user.iloc[i][col]) for col in user.columns
        ]) for i in range(min(len(user), len(user)))]
    )
"""  


if __name__ == "__main__":
    app.run_server(debug=True)






