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

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

from xgboost import XGBClassifier

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
question_to_answer, pp_questions = app_data.get_pp()

ds = dataselector.DataSelector('class')
classifiers = {
    'Decision Tree':                    tree.DecisionTreeClassifier(criterion='entropy', max_depth=5), 
    'Random Forest':                    RandomForestClassifier(n_estimators=10), 
    'XGBoost':                          XGBClassifier(), 
    'Support Vector Machine':           SVC(gamma='auto', probability=True), 
    'Nearest Neighbors':                neighbors.KNeighborsClassifier(), 
    'Naive Bayes':                      GaussianNB()
}

layout = [
    html.Div(children=[
        html.Div(children=[
            html.Button('Data Selector', id='class____open_selector', n_clicks_timestamp=0, style={'margin': '20px'}),

            html.Div(children=[
                html.H5('Label', style={'marginBottom': '0rem'}),
                html.H4(id='label', style={'margin':0})
            ], style={'paddingLeft': '2rem', 'flex': '1 1 auto'}),

            html.Div(children=[
                html.H5('Classifier', style={'marginBottom': '0rem'}),
                dcc.Dropdown(id='classifier',
                    options=[
                        {'label': x, 'value': x} for x in classifiers.keys()
                    ],
                    value='Decision Tree'
                )
            ], style={'width': '15%'}),
            html.Div(children=[
                html.H5('Train/Test', style={'marginBottom': '0rem'}),
                dcc.Slider(
                    id='train_test_split',
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
                html.H5('Accuracy', style={'marginBottom': '0rem'}),
                html.H1('--', id='accuracy')
            ], style={'width': '30%', 'paddingLeft': '2rem'}),

            html.Div(children=[
                html.H5('AUC', style={'marginBottom': '0rem'}),
                html.H1('--', id='auc')
            ], style={'width': '30%', 'paddingLeft': '2rem'}),

            html.Div(children=[
                html.H5('F1', style={'marginBottom': '0rem'}),
                html.H1('--', id='f1')
            ], style={'width': '30%', 'paddingLeft': '2rem'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '2rem', 'paddingBottom': '0'}),
        html.Div(children=[
            html.Div(children=[
                html.H5('Confusion Matrix', style={'marginBottom': '0rem'}),
                html.Div(
                    dcc.Graph(id='confusion_matrix', style={'height': '100%'}),
                    style={'flex': '1 1 auto'}
                )
            ], style={'width': '48%', 'textAlign':'center', 'display':'flex', 'flexDirection': 'column'}),
            html.Div(children=[
                html.H5('Metrics by Label', style={'marginBottom': '0rem'}),
                html.Div(
                    dash_table.DataTable(
                        id='metrics_by_label',
                        columns=[{"name": x, "id": x} for x in ['Label', 'Precision', 'Recall', 'FScore', 'Support']]
                    ),
                    style={'flex': '1 1 auto'}
                )
            ], style={'width': '48%', 'textAlign':'center', 'display':'flex', 'flexDirection': 'column'}),

        ], style={'flex': '1 1 auto', 'display': 'flex', 'justifyContent': 'space-between', 'margin':'0 4rem'})
    ], style={'height': '83vh', 'width': '100vw', 'display': 'flex', 'flexDirection': 'column'}),

    dcc.Store(id='metrics', storage_type='memory'),

    
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

@app.callback(Output('label', 'children'), [Input('class____label', 'value')])
def update_label(value):
    if not value:
        return 'Define in Data Selector'
    return value

@app.callback(Output('class____label', 'options'), [Input('class____all_selected_data', 'data')])
def update_label_options(all_selected_data):
    if not all_selected_data:
        return []

    data = pd.DataFrame.from_dict(all_selected_data['data'])
    
    features = data.columns[1:]

    options = []
    for feature in features:
        if len(data[feature].unique())==2:
            options.append({'label': feature, 'value': feature})

    return options

@app.callback(Output('metrics', 'data'), [Input('classifier', 'value'), Input('train_test_split', 'value'), Input('class____label', 'value'), Input('class____all_selected_data', 'data')])
def update_metrics(classifier, split, label, all_selected_data):
    if not classifier or not split or not label or not all_selected_data:
        return None

    data = pd.DataFrame.from_dict(all_selected_data['data'])
    data = data.applymap(clean_data)

    X = data.drop([label, 'student'], axis=1)
    y = data[[label]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - (split/100.0), random_state = 100)

    model = classifiers[classifier]


    model.fit(X_train, y_train.values.ravel())
    preds = model.predict(X_test)

    try:
        prob_preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob_preds)
    except IndexError:
        auc = 'Could not calculate'

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='micro')
    confusion = confusion_matrix(y_test, preds)

    precision, recall, fscore, support = score(y_test, preds)
    by_label = {'Precision': precision, 'Recall': recall, 'FScore': fscore, 'Support': support}

    return {
        'accuracy': accuracy, 
        'auc': auc, 
        'f1': f1, 
        'confusion': confusion,
        'by_label': by_label
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

for x in [('accuracy', True), ('auc', False), ('f1', False)]:
    app.callback(Output(x[0], 'children'), [Input('metrics', 'data')])(generate_update_metric_callback(x[0], x[1]))

@app.callback(Output('confusion_matrix', 'figure'), [Input('metrics', 'data')])
def update_confusion_matrix(data):
    if not data:
        return go.Figure()

    matrix = data['confusion']
    matrix = list(reversed(matrix))
    return go.Figure(
        data=[go.Heatmap(
            z = matrix,
            y = ['Actual: 1', 'Actual: 0'],
            x = ['Predicted: 0', 'Predicted: 1']
        )],
        layout=go.Layout(
            margin=go.layout.Margin(
                t=10,
            )
        )
    )

@app.callback(Output('metrics_by_label', 'data'), [Input('metrics', 'data')])
def update_metrics_table(data):
    if not data:
        return []

    by_label = data['by_label']
    by_label['Label'] = [0, 1]
    df = pd.DataFrame(by_label)

    df = df.applymap(prettify_data)

    return df.to_dict('records')
    