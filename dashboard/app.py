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
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px


import pickle
import shap

"""-------Chargement données et modèle-------"""
application_train = pd.read_csv('outputs/application_train_reduced.csv')
infos_clients = application_train.columns[:10]

df = pd.read_csv('outputs/df_reduced.csv', nrows=50)
model = pickle.load(open('outputs/lgbm_model.sav', 'rb'))

drop_col =['SK_ID_CURR', 'TARGET', 'PREDICTIONS']
dataframe=df.drop(columns=drop_col)
explainer = shap.TreeExplainer(model)

seuil_score = 0.20

class ShapObject:
    
    def __init__(self, base_values, data, values, feature_names):
        self.base_values = base_values # Single value
        self.data = data # Raw feature values for 1 row of data
        self.values = values # SHAP values for the same row of data
        self.feature_names = feature_names # Column names

"""-------Dash parameters-------"""
left_col_width = 3
#right_col_width = "100%"

"""-------Dash app-------"""

app = dash.Dash(
    #external_stylesheets=[dbc.themes.SLATE]
    #external_stylesheets=[dbc.themes.BOOTSTRAP]
    external_stylesheets=[dbc.themes.CERULEAN]

    )

app.layout = dbc.Container([
    
    dcc.Store(id='important-feat'),
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
            html.H4("Client sélectionné"),
 
            dcc.Dropdown(id='user-id-dd',                         
                         options=[{'label': i, 'value': i} for i in df['SK_ID_CURR']],
                         placeholder="Sélectionnez un client"
                                     )
                  
            ],width=left_col_width ), # end first col
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    # Card 1
                    dbc.Card(id='card1',
                             children= [
                        
                        dbc.CardBody(
                            [
                                html.H6("Prédiction", className="card-title"),
                                html.P(
                                    'Contenu card',
                                    className='card-text',
                                    id='card-pred'
                                    )
                                ]
                            )                                              
                        ]),
                    ]),

                dbc.Col([                    
                    # Card 2
                    dbc.Card(children= [
                        
                        dbc.CardBody(
                            [
                                html.H6(f"Score client (cible = {seuil_score})", className="card-title"),                                
                                html.P(
                                    'Afficher score',
                                    className='card-text',
                                    id='card-score'
                                    )
                                ]
                            #, style = {'display': 'none'}
                            )                                              
                        ]),  
                    ]),
                
                dbc.Col([ 
                    # Card 3
                     dbc.Card(children= [
                        
                        dbc.CardBody(
                            [
                                html.H6("Montant crédit", className="card-title"),
                                html.P(
                                    'Afficher montant',
                                    className='card-text',
                                    id='card-amt'
                                    )
                                ]
                            #, style = {'display': 'none'}
                            )                                              
                        ])                  
                    ])
                               
                ])
            
                     
            ], ) # end 2nd col

        ]), # end 2nd row
    
    html.Br(),

    #3 row
    dbc.Row([
        dbc.Col([
            
            #Datatable infos clients
            #dbc.Card("Datatable infos clients", body=True)
            html.Div(id='table')
                     
           ], width=left_col_width), #end 1st col
        
        dbc.Col([
            
            # Graphs
            #dbc.Card("Graphs", body=True)  
            dbc.Tabs(
                [
                    dbc.Tab(label="Client", tab_id="client"),
                    dbc.Tab(label="Group", tab_id="group"),
                ],
                id="tabs",
                active_tab="client",
            ),
            html.Div(id="client-tab", className="p-4", children= [
                dbc.Row(
                    [
                        #add dbc options,
                        dbc.Col(dcc.Graph(id='wf-graph', figure={})),
                    ]
                )
            ], style = {'display': 'none'}), 

            html.Div(id="group-tab", className="p-4", children= [
                dbc.Row(
                    [dbc.Row([
                        #
                        dbc.Col(dcc.Dropdown(id='dd-group',                         
                                     options=[{'label': i, 'value': i} for i in dataframe.columns],
                                     value=['EXT_SOURCES_MEAN', 'EXT_SOURCE_3', 'AMT_ANNUITY', 'CREDIT_TO_GOODS_RATIO'],
                                     multi=True
                                                 ))
                        ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='hist', figure={})),
                        #
                        ])
                    ]
                )
            ]) 
                   
            ]) # end 2nd col
        
        ]), # end 3rd row
    
    html.Br(),

    #4 row
    dbc.Row([
        dbc.Col([
            
            #Annotations
            dbc.Card(
                [
                    dbc.CardHeader(html.H6("Observations", className="card-title")),
                    dbc.CardBody(
                        [
                            html.Div([dcc.Textarea()])
                            ]
                        )
                    ]
                )
                       
            ], width=left_col_width), #end 1st col
        
        dbc.Col([
            
            # Text explications
            dbc.Card(
                [
                    dbc.CardHeader(html.H6("Explicabilité du modèle (définitons des variables ciblées)", className="card-title")),
                    dbc.CardBody(
                        [
                            html.Div(id='table-explication')
                            ]
                        )
                    ]
                )
            ])
        
        ]) # end 4th row       

    
    ]) # end layout container




@app.callback(
    Output('table', 'children'),
    Input('user-id-dd', 'value')
    )
def update_info_table(user_id):
    if user_id != None:
        user = application_train[infos_clients][application_train['SK_ID_CURR']==user_id].transpose().reset_index()
        user.columns = ['Infos', 'Client']
    else:
        user = pd.DataFrame(columns=['Infos', 'Client'])
    #return dbc.Table.from_dataframe(user)
    return dbc.Table.from_dataframe(user, striped=True, bordered=True, hover=True)

@app.callback(
    [Output('card-pred', 'children'),
     Output('card1', 'color'),
     Output('card1', 'inverse'),
     Output('card-score', 'children'), 
     Output('card-amt', 'children')],
     Input('user-id-dd', 'value')
    )
def update_cards(user_id):
    """
    Update cards
    
    """
    if user_id != None:
        amt_loan = df[df['SK_ID_CURR']==user_id]['AMT_CREDIT'].reset_index(drop=True)[0]
        score = round(df[df['SK_ID_CURR']==user_id]['PREDICTIONS'].reset_index(drop=True)[0],2)

        if score <= seuil_score:
            color ='green'
            prediction = 'Crédit favorable'
        #elif score <= seuil_score:
        else:
            color='red'
            prediction = 'Crédit à risque'            
        return prediction, color, True, score.astype(str), amt_loan
    else:
        return '', '', '', '', ''
            


@app.callback(
    [Output("client-tab", "style"), Output("group-tab", "style")],
    [Input("tabs", "active_tab")],
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    on = {'display': 'block'}
    off = {'display': 'none'}
    if active_tab is not None:
        if active_tab == "group":
            return off, on
        elif active_tab == "client":
            return on, off
    return "No tab selected"

@app.callback([Output("wf-graph", "figure"),  
               Output("hist", "figure"),
               Output("important-feat", "data")],
              [Input('user-id-dd', 'value'),
               Input("dd-group", "value")])
def generate_graphs(user_id, selected_values):
    """
    Update graphs
    """
    if user_id == None:
        # generate empty graphs when app loads
        return {}, {}, {}

    elif user_id != None:
        row = df[df['SK_ID_CURR']==user_id].index
        x_new = dataframe.iloc[row,:]
        # do shap value calculations for basic waterfall plot
        shap_values_client = explainer.shap_values(x_new)
        updated_fnames = x_new.T.reset_index()
        updated_fnames.columns = ['feature', 'value']
        updated_fnames['shap_original'] = pd.Series(-shap_values_client[0][0])
        updated_fnames['shap_abs'] = updated_fnames['shap_original'].abs()
        updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)
        important_features=updated_fnames.sort_values(by=['shap_abs'], ascending=False)['feature'][:5]
        important_features = pd.DataFrame({'feature':important_features}).to_dict('records')
    
        # need to collapse those after first 9, so plot always shows 10 bars
        show_features = 9
        num_other_features = updated_fnames.shape[0] - show_features
        col_other_name = f"{num_other_features} other features"
        f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
        f_group['feature'] = col_other_name
        plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])
    
        # additional things for plotting
        plot_range = plot_data['shap_original'].cumsum().max() - plot_data['shap_original'].cumsum().min()
        plot_data['text_pos'] = np.where(plot_data['shap_original'].abs() > (1/9)*plot_range, "inside", "outside")
        plot_data['text_col'] = "white"
        plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] < 0), 'text_col'] = "#3283FE"
        plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] > 0), 'text_col'] = "#F6222E"
        
        
        fig1 = go.Figure(go.Waterfall(
            name="",
            orientation="h",
            measure=['absolute'] + ['relative']*show_features,
            base=explainer.expected_value[0],
            textposition=plot_data['text_pos'],
            text=plot_data['shap_original'],
            textfont={"color": plot_data['text_col']},
            texttemplate='%{text:+.2f}',
            y=plot_data['feature'],
            x=plot_data['shap_original'],
            connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
            decreasing={"marker": {"color": "#3283FE"}},
            increasing={"marker": {"color": "#F6222E"}},
            hoverinfo="skip"
        ))
        fig1.update_layout(
            waterfallgap=0.2,
            autosize=False,
            width=770,
            height=570,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                showgrid=True,
                zeroline=True,
                showline=True,
                gridcolor='lightgray'
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True,
                showticklabels=True,
                linecolor='black',
                tickcolor='black',
                ticks='outside',
                ticklen=5
            ),
            #margin={'t': 25, 'b': 50},
            #margin={'l': 15},

            shapes=[
                dict(
                    type='line',
                    yref='paper', y0=0, y1=1.02,
                    xref='x', x0=plot_data['shap_original'].sum()+explainer.expected_value[0],
                    x1=plot_data['shap_original'].sum()+explainer.expected_value[0],
                    layer="below",
                    line=dict(
                        color="black",
                        width=1,
                        dash="dot")
                )
            ]
        )
        fig1.update_yaxes(automargin=True)
        fig1.add_annotation(
            yref='paper',
            xref='x',
            x=explainer.expected_value[0],
            y=-0.12,
            text="E[f(x)] = {:.2f}".format(explainer.expected_value[0]),
            showarrow=False,
            font=dict(color="black", size=14)
        )
        fig1.add_annotation(
            yref='paper',
            xref='x',
            x=plot_data['shap_original'].sum()+explainer.expected_value[0],
            y=1.075,
            text="f(x) = {:.2f}".format(plot_data['shap_original'].sum()+explainer.expected_value[0]),
            showarrow=False,
            font=dict(color="black", size=14)
        )
        
        # Histogrammes
        n_rows = len(selected_values) // 2 + 1
        fig2 = make_subplots(rows=n_rows, cols=2)
        
        for n in range(len(selected_values)):
            name = selected_values[n]
            fig2.add_trace(
                go.Histogram(x=df[name], name=name),
            row=n//2+1, col=n%2+1
        )
        fig2.update_layout(width=770, 
                           height=570,
                           legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                                ))
        return fig1, fig2, important_features
    else:
        return {}, {}, {}
 
@app.callback(
    Output('table-explication', 'children'),
    Input('important-feat', 'data')
    )
def update_explanation_table(important_feat):
    """
    Remplissage des définitions des features principales

    """
    table = pd.DataFrame(important_feat)
    table['Définition'] = ""  # Ici remplir avec les définitions
    return dbc.Table.from_dataframe(table, striped=True, bordered=True, hover=True)
 
# =============================================================================
# @app.callback(
#     Output("dd-group", "options"), 
#     Input("dd-group", "value"),
# )
# def limit_drop_options(selected_values):
#     """Limit histogram dropdown to at most 4 actives selections"""
#     if len(selected_values) > 4:
#         return selected_values[:4]
#     else:
#         return dataframe.columns
# 
# =============================================================================
# =============================================================================
# @app.callback(
#     Output('tab-content', 'children'),
#     [Input('user-id-dd', 'value'), Input('tabs', 'active_tab')]
#     )
# def update_graph(user_id, active_tab):
#         if user_id and active_tab is not None:
#             if active_tab == 'client':
#                 row = df[df['SK_ID_CURR']==user_id].index
#                 return dbc.Card(f"row = {row}", body=True)
#             elif active_tab == 'groupe':
#                 return dbc.Card("tab2", body=True)
#             else:
#                 return "No tab selected"
#         else:
#             return "No tab selected"
# =============================================================================

# =============================================================================
# @app.callback(
#     Output("tab-content", "children"),
#     [Input("tabs", "active_tab"), Input('user-id-dd', 'value')]
# )
# =============================================================================
# =============================================================================
# def render_tab_content(active_tab, user_id):
#     if user_id != None:
#         if active_tab is not None:
#             if active_tab == "tab-client":
#                 row = df[df['SK_ID_CURR']==user_id].index[0]
#                 shap_values = explainer(dataframe.values)
#                 shap_object = ShapObject(base_values = explainer.expected_value[1],
#                                          values = explainer.shap_values(dataframe)[1][row,:],
#                                          feature_names = dataframe.columns,
#                                          data = dataframe.iloc[row,:])            
#                 return dcc.Graph(
#                     figure=shap.waterfall_plot(shap_object))
#             elif active_tab == "tab-group":
#                 return dbc.Card("tab2", body=True)
#     return "No tab selected"
# =============================================================================
# =============================================================================
# def render_tab_content(active_tab, user_id):
#     if user_id != None:
#         if active_tab is not None:
#             if active_tab == "tab-client":
#                 row = df[df['SK_ID_CURR']==user_id].index[0]
#                 shap_values = explainer(dataframe.values)
#                 shap_object = ShapObject(base_values = explainer.expected_value[1],
#                                          values = explainer.shap_values(dataframe)[1][row,:],
#                                          feature_names = dataframe.columns,
#                                          data = dataframe.iloc[row,:])
#                 
#                 waterfall_plot = shap.waterfall_plot(shap_object, matplotlib=False)
#                 shap_html = f"<head>{shap.getjs()}</head><body>{waterfall_plot.html()}</body>"
#                 return html.Iframe(srcDoc=shap_html,
#                        style={"width": "100%", "height": "200px", "border": 0})
#             elif active_tab == "tab-group":
#                 return dbc.Card("tab2", body=True)
#     return "No tab selected"
# =============================================================================
# =============================================================================
# def render_tab_content(active_tab, user_id):
#     if user_id != None:
#         if active_tab is not None:
#             if active_tab == "tab-client":
#                 row = df[df['SK_ID_CURR']==user_id].index[0]
#                 shap_values = explainer(dataframe.values)
#                 shap_object = ShapObject(base_values = explainer.expected_value[1],
#                                          values = explainer.shap_values(dataframe)[1][row,:],
#                                          feature_names = dataframe.columns,
#                                          data = dataframe.iloc[row,:])
#                 
#                 force_plot = shap.force_plot(shap_object, matplotlib=False)
#                 shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
#                 return html.Iframe(srcDoc=shap_html,
#                        style={"width": "100%", "height": "200px", "border": 0})
#             elif active_tab == "tab-group":
#                 return dbc.Card("tab2", body=True)
#     return "No tab selected"
# =============================================================================


# =============================================================================
# def _force_plot_html(*args):
#     force_plot = shap.force_plot(*args, matplotlib=False)
#     shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
#     return html.Iframe(srcDoc=shap_html,
#                        style={"width": "100%", "height": "200px", "border": 0})
# =============================================================================


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
    
    
# =============================================================================
# row = df[df['SK_ID_CURR']==100002].index
# x_new = dataframe.iloc[row,:]
# # do shap value calculations for basic waterfall plot
# shap_values_client = explainer.shap_values(x_new)
# updated_fnames = x_new.T.reset_index()
# updated_fnames.columns = ['feature', 'value']
# updated_fnames['shap_original'] = pd.Series(-shap_values_client[0][0])
# =============================================================================

# =============================================================================
# liste=['a','b','c','d']
# dictio=pd.DataFrame({'feature':liste}).to_dict('records')
# dff = pd.DataFrame(dictio)
# dffd = pd.DataFrame({})
# =============================================================================
