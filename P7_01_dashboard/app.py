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
# import dash_table_experiments as dt
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
infos_clients = application_train.columns[:14]

df = pd.read_csv('outputs/df_reduced.csv', nrows=50)
model = pickle.load(open('outputs/lgbm_model.sav', 'rb'))

drop_col = ['SK_ID_CURR', 'TARGET', 'PREDICTIONS']
dataframe = df.drop(columns=drop_col)
explainer = shap.TreeExplainer(model)

seuil_score = 0.20

"""-------Dash parameters-------"""
left_col_width = 3
# right_col_width = "100%"

"""-------Dash app-------"""

app = dash.Dash(
    # external_stylesheets=[dbc.themes.SLATE]
    # external_stylesheets=[dbc.themes.BOOTSTRAP]
    external_stylesheets=[dbc.themes.CERULEAN]
)
server = app.server

app.layout = dbc.Container([

    dcc.Store(id='important-feat'),
    # 1 row
    dbc.Row([
        # Title
        dbc.Col([
            # Title
            html.H1("Client Loan Management",
                    className='text-left, mb-4')
        ], )  # end col
    ]),  # end first row

    # 2 row
    dbc.Row([
        dbc.Col([

            # Dropdown
            # dbc.Card("Dropdown", body=True)
            html.H4("Client sélectionné"),

            dcc.Dropdown(id='user-id-dd',
                         options=[{'label': i, 'value': i} for i in df['SK_ID_CURR']],
                         placeholder="Sélectionnez un client"
                         )

        ], width={'size': left_col_width, 'offset': 0}),  # end first col
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    # Card 1
                    dbc.Card(id='card1',
                             children=[

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
                    dbc.Card(children=[

                        dbc.CardBody(
                            [
                                html.H6(f"Score client (cible = {seuil_score})", className="card-title"),
                                html.P(
                                    'Afficher score',
                                    className='card-text',
                                    id='card-score'
                                )
                            ]
                            # , style = {'display': 'none'}
                        )
                    ]),
                ]),

                dbc.Col([
                    # Card 3
                    dbc.Card(children=[

                        dbc.CardBody(
                            [
                                html.H6("Montant crédit", className="card-title"),
                                html.P(
                                    'Afficher montant',
                                    className='card-text',
                                    id='card-amt'
                                )
                            ]
                            # , style = {'display': 'none'}
                        )
                    ])
                ])

            ])

        ], )  # end 2nd col

    ]),  # end 2nd row

    html.Br(),

    # 3 row
    dbc.Row([
        dbc.Col([

            # Datatable infos clients
            # dbc.Card("Datatable infos clients", body=True)
            html.Div(id='table')

        ], width=left_col_width),  # end 1st col

        dbc.Col([

            # Graphs
            dbc.Card(
                [
                    dbc.CardBody([

                        dbc.Tabs(
                            [
                                dbc.Tab(label="Client", tab_id="client"),
                                dbc.Tab(label="Group", tab_id="group"),
                            ],
                            id="tabs",
                            active_tab="client",
                        ),
                        html.Div(id="client-tab", className="p-4", children=[
                            dbc.Row(
                                [
                                    # add dbc options,
                                    dbc.Col(dcc.Graph(id='wf-graph', figure={})),
                                ]
                            )
                        ], style={'display': 'none'}),

                        html.Div(id="group-tab", className="p-4", children=[
                            dbc.Row(
                                [dbc.Row([
                                    #
                                    dbc.Col(dcc.Dropdown(id='dd-group',
                                                         options=[{'label': i, 'value': i} for i in dataframe.columns],
                                                         value=['EXT_SOURCES_MEAN', 'EXT_SOURCE_3', 'AMT_ANNUITY',
                                                                'CREDIT_TO_GOODS_RATIO'],
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
                    ])
                ])
        ])  # end 2nd col

    ]),  # end 3rd row

    html.Br(),

    # 4 row
    dbc.Row([
        dbc.Col([

            # Annotations
            dbc.Card(
                [
                    dbc.CardHeader(html.H6("Observations", className="card-title")),
                    dbc.CardBody(
                        [
                            html.Div([dcc.Textarea(style={'width': '100%', 'height': 280})])
                        ], style={'height': '100%'}
                    )
                ], style={'height': '100%'}
            )

        ], width=left_col_width),  # end 1st col

        dbc.Col([

            # Text explications
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.H6("Explicabilité du modèle (définitons des variables ciblées)", className="card-title")),
                    dbc.CardBody(
                        [
                            html.Div(id='table-explication')
                        ]
                    )
                ]
            )
        ])

    ])  # end 4th row

], fluid=True
    # style={ 'width': "100vw"}
)  # end layout container


@app.callback(
    Output('table', 'children'),
    Input('user-id-dd', 'value')
)
def update_info_table(user_id):
    """
    Ce callback prend la valeur de l'identifiant en entrée et met à jour la table d'infos client
    """
    if user_id != None:
        user = application_train[infos_clients][application_train['SK_ID_CURR'] == user_id].transpose().reset_index()
        user.columns = ['Infos', 'Client']
    else:
        user = pd.DataFrame(columns=['Infos', 'Client'])
    # return dbc.Table.from_dataframe(user)
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
    Ce callback met à jour les cartes prédiction, score client,
    montant du crédit en fonction de l'identifiant client sélectionné

    """
    if user_id != None:
        amt_loan = df[df['SK_ID_CURR'] == user_id]['AMT_CREDIT'].reset_index(drop=True)[0]
        score = round(df[df['SK_ID_CURR'] == user_id]['PREDICTIONS'].reset_index(drop=True)[0], 2)

        if score <= seuil_score:
            color = 'green'
            prediction = 'Crédit favorable'
        # elif score <= seuil_score:
        else:
            color = 'red'
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
    Ce callback agit comme un interrupteur qui permet d'afficher
    soit l'onglet Client, soit l'onglet Group
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
    Ce callback met à jour les différents graphes, il renvoit également les 5 features les plus importantes dans
    l'attribution du score client, en vue d'expliquer ce dernier
    """
    if user_id == None:
        # generate empty graphs when app loads
        return {}, {}, {}

    elif user_id != None:
        row = df[df['SK_ID_CURR'] == user_id].index
        x_new = dataframe.iloc[row, :]
        # do shap value calculations for basic waterfall plot
        shap_values_client = explainer.shap_values(x_new)
        updated_fnames = x_new.T.reset_index()
        updated_fnames.columns = ['feature', 'value']
        updated_fnames['shap_original'] = pd.Series(-shap_values_client[0][0])
        updated_fnames['shap_abs'] = updated_fnames['shap_original'].abs()
        updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)
        important_features = updated_fnames.sort_values(by=['shap_abs'], ascending=False)['feature'][:5]
        important_features = pd.DataFrame({'feature': important_features}).to_dict('records')

        # need to collapse those after first 9, so plot always shows 10 bars
        show_features = 9
        num_other_features = updated_fnames.shape[0] - show_features
        col_other_name = f"{num_other_features} other features"
        f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
        f_group['feature'] = col_other_name
        plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])

        # additional things for plotting
        plot_range = plot_data['shap_original'].cumsum().max() - plot_data['shap_original'].cumsum().min()
        plot_data['text_pos'] = np.where(plot_data['shap_original'].abs() > (1 / 9) * plot_range, "inside", "outside")
        plot_data['text_col'] = "white"
        plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] < 0), 'text_col'] = "#3283FE"
        plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] > 0), 'text_col'] = "#F6222E"

        fig1 = go.Figure(go.Waterfall(
            name="",
            orientation="h",
            measure=['absolute'] + ['relative'] * show_features,
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
            # autosize=True,
            width=1100,
            height=580,
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
            # margin={'t': 25, 'b': 50},
            # margin={'l': 15},

            shapes=[
                dict(
                    type='line',
                    yref='paper', y0=0, y1=1.02,
                    xref='x', x0=plot_data['shap_original'].sum() + explainer.expected_value[0],
                    x1=plot_data['shap_original'].sum() + explainer.expected_value[0],
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
            x=plot_data['shap_original'].sum() + explainer.expected_value[0],
            y=1.075,
            text="f(x) = {:.2f}".format(plot_data['shap_original'].sum() + explainer.expected_value[0]),
            showarrow=False,
            font=dict(color="black", size=14)
        )

        # Histogrammes
        n_rows = (len(selected_values) - 1) // 2 + 1
        fig2 = make_subplots(rows=n_rows, cols=2)

        for n in range(len(selected_values)):
            name = selected_values[n]
            row = n // 2 + 1
            col = n % 2 + 1

            # Histogrammes pour target = 0 et target = 1
            fig2.add_trace(
                go.Histogram(x=df[df['TARGET'] == 0][name], name=name),
                # marker = dict(color = 'green'),
                row=row, col=col
            )
            fig2.add_trace(
                go.Histogram(x=df[df['TARGET'] == 1][name], name=name),
                # marker_color='#EB89B5',
                # marker = dict(color = 'red'),
                row=row, col=col
            )

            x = df[df['SK_ID_CURR'] == user_id][name].reset_index(drop=True)[0]
            x0 = df[df['TARGET'] == 0][name].mean()
            x1 = df[df['TARGET'] == 1][name].mean()
            # Ligne client
            fig2.add_shape(
                go.layout.Shape(type='line', xref='x', yref='y domain',
                                x0=x, y0=0, x1=x, y1=0.9, line={'dash': 'dash'}),
                row=row, col=col
            )
            # Ligne moyenne pour Target = 0 (clients en règle)
            fig2.add_shape(
                go.layout.Shape(type='line', xref='x', yref='y domain',
                                x0=x0, y0=0, x1=x0, y1=0.8, line={'dash': 'dash', 'color': 'green'}),
                row=row, col=col
            )
            # Ligne moyenne pour target = 1 (clients à risque)
            fig2.add_shape(
                go.layout.Shape(type='line', xref='x', yref='y domain',
                                x0=x1, y0=0, x1=x1, y1=0.8, line={'dash': 'dash', 'color': 'red'}),
                row=row, col=col
            )

            # Annotations
            fig2.add_annotation(x=x, y=0.95,
                                yref='y domain',
                                text="Client",
                                showarrow=False,
                                yshift=10,
                                row=row, col=col)
            fig2.add_annotation(x=x0, y=0.85,
                                yref='y domain',
                                text="Moy. En règle",
                                showarrow=False,
                                yshift=10,
                                row=row, col=col,
                                font=dict(
                                    color="green",
                                    size=12)
                                ),
            fig2.add_annotation(x=x1, y=0.85,
                                yref='y domain',
                                text="Moy. En défaut",
                                showarrow=False,
                                yshift=10,
                                row=row, col=col,
                                font=dict(
                                    color="red",
                                    size=12))
            # Layout du graphe 2
        fig2.update_layout(
            # autosize=True,
            barmode='overlay',
            width=1250,
            height=580,
            margin=dict(l=0, r=0, t=50, b=0),
            # =============================================================================
            #                            legend=dict(
            #                                 orientation="h",
            #                                 yanchor="bottom",
            #                                 y=1.02,
            #                                 xanchor="right",
            #                                 x=1
            #                                 )
            # =============================================================================
        )
        fig2.update_traces(opacity=0.6)
        return fig1, fig2, important_features
    else:
        return {}, {}, {}


@app.callback(
    Output('table-explication', 'children'),
    Input('important-feat', 'data')
)
def update_explanation_table(important_feat):
    """
    Ce callback prend en entrée les features importantes pour l'explicabilité du score client afin d'afficher
    leur définition dans une table (les définitions sont à ajouter)

    """
    table = pd.DataFrame(important_feat)
    table['Définition'] = ""  # Ici remplir avec les définitions
    return dbc.Table.from_dataframe(table, striped=True, bordered=True, hover=True)


if __name__ == "__main__":
    app.run_server(debug=True)
