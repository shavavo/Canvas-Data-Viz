from app import dash_app as app
from app import tab_selected_style, tab_style, tabs_styles
import dash

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

from pages import clustering, dimred, classification, regression


layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    
    dcc.Tabs(id="tabs-master", value='tab-1', children=[
        dcc.Tab(label='Dimensionality Reduction', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Clustering', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Classification', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Regression', value='tab-4', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),

    html.Div(id='content')
])

@app.callback(Output('url', 'pathname'), [Input('tabs-master', 'value')])
def update_path(value):
    if value=='tab-2':
        return '/predictive/clustering'
    if value=='tab-3':
        return '/predictive/classification'
    if value=='tab-4':
        return '/predictive/regression'
    
    return '/predictive/dimred'

@app.callback(Output('content', 'children'), [Input('url', 'pathname')])
def update_page(path):
    if path=='/predictive/dimred':
        return dimred.serve_layout()
    if path=='/predictive/clustering':
        return clustering.layout
    if path=='/predictive/classification':
        return classification.serve_layout()
    if path=='/predictive/regression':
        return regression.serve_layout()
    
    return []