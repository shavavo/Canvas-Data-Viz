from app import dash_app as app
from app import app as flask_app
from app import tab_selected_style, tab_style

import visdcc

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import uuid

import app_data_manager as app_data
from pages import dataselector

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
question_to_answer, pp_questions = app_data.get_pp()

ds = dataselector.DataSelector('reg')


layout = [
    html.Div(children=[
        html.Div(children=[
            html.Button('Data Selector', id='reg____open_selector', n_clicks_timestamp=0, style={'margin': '20px'}),

            html.Div(children=[
                html.H5('Label', style={'marginBottom': '0rem'}),
                html.H4(id='reg_label', style={'margin':0})
            ], style={'paddingLeft': '2rem', 'flex': '1 1 auto'}),

            html.Div(children=[
                # html.H5('Classifier', style={'marginBottom': '0rem'}),
                # dcc.Dropdown(id='classifier',
                #     options=[
                #         {'label': x, 'value': x} for x in classifiers.keys()
                #     ],
                #     value='Decision Tree'
                # )
            ], style={'width': '15%'}),
            html.Div(children=[
                html.H5('Train/Test', style={'marginBottom': '0rem'}),
                dcc.Slider(
                    id='reg_train_test_split',
                    min=50,
                    max=80,
                    step=5,
                    # dots=True,
                    marks={
                        50: '50% Train',
                        60: '60%',
                        70: '70%',
                        80: '80% Train',
                    },
                    value=70,
                    className='paddingLeft'
                )
            ], style={'width': '15%', 'paddingLeft': '2rem', 'paddingRight': '5rem'}),
        ], style={'width': '100%', 'display': 'flex'})
    ], style={'width': '100%', 'height':'12vh', 'display': 'flex', 'align-items': 'center'}),

    html.Hr(style={'margin': '0'}),

    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.H5('R2', style={'marginBottom': '0rem'}),
                html.H1('--', id='r2')
            ], style={'width': '30%', 'paddingLeft': '2rem'}),

            html.Div(children=[
                html.H5('RMSE', style={'marginBottom': '0rem'}),
                html.H1('--', id='rmse')
            ], style={'width': '30%', 'paddingLeft': '2rem'}),

            html.Div(children=[
                html.H5('MAE', style={'marginBottom': '0rem'}),
                html.H1('--', id='mae')
            ], style={'width': '30%', 'paddingLeft': '2rem'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '2rem', 'paddingBottom': '0'}),
        html.Div(children=[
            # html.Div(children=[
            #     html.H5('Confusion Matrix', style={'marginBottom': '0rem'}),
            #     # html.Div(
            #     #     dcc.Graph(id='confusion_matrix', style={'height': '100%'}),
            #     #     style={'flex': '1 1 auto'}
            #     # )
            # ], style={'width': '48%', 'textAlign':'center', 'display':'flex', 'flexDirection': 'column'}),
            html.Div(children=[
                html.H5('Coefficients', style={'marginBottom': '0rem'}),
                html.Div(
                    dash_table.DataTable(
                        id='coefficients',
                        columns=[{"name": x, "id": x} for x in ['Feature', 'Coefficient']],
                        style_table={
                            'maxHeight': '400px',
                            'overflowY': 'scroll'
                        },
                    ),
                    style={'flex': '1 1 auto'}
                )
            ], style={'width': '48%', 'textAlign':'center', 'display':'flex', 'flexDirection': 'column'}),

        ], style={'flex': '1 1 auto', 'display': 'flex', 'justifyContent': 'space-between', 'margin':'0 4rem'})
    ], style={'height': '83vh', 'width': '100vw', 'display': 'flex', 'flexDirection': 'column'}),

    dcc.Store(id='reg_metrics', storage_type='memory'),

    
]

def serve_layout():
    session_id = str(uuid.uuid4())
    layout.append(ds.serve_layout(session_id))
    return layout

def clean_data(x):
    if isinstance(x, str):
        return int(x.split('-')[0])
    return x

def prettify_data(x):
    return round(x, 3)


# @app.callback(Output('label', 'children'), [Input('class____label', 'value')])

@app.callback(Output('reg____label', 'options'), [Input('reg____all_selected_data', 'data')])
def update_label_options(all_selected_data):
    """
        Restricts label selector in dataselector to continuous variables
    """
    if not all_selected_data:
        return []

    data = pd.DataFrame.from_dict(all_selected_data['data'])
    
    features = data.columns[1:]

    options = []
    for feature in features:
        if len(data[feature].unique()) > 2:
            options.append({'label': feature, 'value': feature})

    return options

@app.callback(Output('reg_label', 'children'), [Input('reg____label', 'value')])
def update_label(value):
    if not value:
        return 'Define in Data Selector'
    return value

@app.callback(Output('reg_metrics', 'data'), [Input('reg_train_test_split', 'value'), Input('reg____label', 'value'), Input('reg____all_selected_data', 'data')])
def update_metrics(split, label, all_selected_data):
    if not split or not label or not all_selected_data:
        return None

    data = pd.DataFrame.from_dict(all_selected_data['data'])
    data = data.applymap(clean_data)

    X = data.drop([label, 'student'], axis=1)
    y = data[[label]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - (split/100.0), random_state = 100)

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    r_sq = linreg.score(X_train, y_train)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    return {
        'r2':       r_sq,
        'mae':      mae,
        'rmse':     rmse,
        'intercept':    linreg.intercept_,
        'coef':         linreg.coef_[0],
        'features':     X.columns
    }

def generate_update_metric_callback(element, percent):
    def update_metric(data):
        if not data:
            return '--'
        
        if isinstance(data[element], str):
            return data[element]

        if percent:
            return str(round(data[element]*100, 2)) + '%'

        return round(data[element], 3)
    
    return update_metric

for x in [('r2', False), ('mae', False), ('rmse', False)]:
    app.callback(Output(x[0], 'children'), [Input('reg_metrics', 'data')])(generate_update_metric_callback(x[0], x[1]))

@app.callback(Output('coefficients', 'data'), [Input('reg_metrics', 'data')])
def update_metrics_table(data):
    if not data:
        return []

    coef_data = []

    features = data['features']
    coefs = data['coef']


    for column, coef in zip(features, coefs):
        coef_data.append({'Feature': column, 'Coefficient': prettify_data(coef)})
   
    df = pd.DataFrame(coef_data)


    return df.to_dict('records')