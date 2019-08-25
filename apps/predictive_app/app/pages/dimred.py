from app import dash_app as app
from app import app as flask_app
from app import tab_selected_style, tab_style

import visdcc

import dash
import re

import flask
import os

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd


import app_data_manager as app_data

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples

from scipy.cluster.hierarchy import dendrogram, linkage
from plotly import figure_factory as FF
from plotly import tools

import numpy as np
import uuid

from pages import dataselector

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
question_to_answer, pp_questions = app_data.get_pp()

ds = dataselector.DataSelector('dimred')

def clean_columns(data):
    features = data.columns[1:]
    for feature in features:
        if re.search(r'^Q\d{1,3}:', feature) or ('Concentration:' in feature) or ('Company:' in feature):
            data[feature] = data[feature].str.replace(r'\D+', '').astype('int')
    return data

def word_wrap(text, length):
    split = text.split()
    result = ""
    current = 0
    for word in split:
        current += len(word)
        
        if current > length:
            result += "<br />"
            current = 0
        
        result += word + " "
    return result


layout = [
    dcc.Store(id='pca_data', storage_type='memory'),
    dcc.Store(id='km_data', storage_type='memory'),
    dcc.Store(id='hc_data', storage_type='memory'),

    html.Div(children=[
        html.Div(children=[
            html.Button('Data Selector', id='dimred____open_selector', n_clicks_timestamp=0, style={'margin': '20px'}),

            html.Div(children=[
                html.H5('Coloring', style={'marginBottom': '0rem'}),
                dcc.Dropdown(
                    options=[
                        {'label': 'None', 'value': 'None'},
                        {'label': 'K-means', 'value': 'KM'},
                        {'label': 'Hierarchical Clustering', 'value': 'HC'}
                    ],
                    value='None',
                    id='coloring_dd'
                ),
            ], style={'width': '20%'}),

            html.Div(
                children= [
                    html.Div(
                        id='dimred_clusters',
                        children=[
                            html.H5('Clusters:', style={'marginBottom': '0rem'}),
                            dcc.Slider(
                                min=2,
                                max=8,
                                step=1,
                                value=2,
                                marks={2: '2', 4:'4', 6:'6', 8:'8'},
                                id='dimred_clusters-slider',
                                className='margin-1rem ', 
                            )
                        ], 
                        style={'width': '20%', 'display': 'none'}
                    ),
                    html.Div(
                        id='dimred_show',
                        children=[
                            html.H5('Show:', style={'marginBottom': '0rem'}),
                            dcc.Checklist(
                                id='show_cl',
                                options=[
                                    {'label': 'Centroids', 'value': 'Centroids'},
                                    {'label': 'Cluster Shapes', 'value': 'Cluster Shapes'}
                                ],
                                value=['Centroids']
                            )
                        ], 
                        style={'width': '30%', 'marginLeft': '40px', 'display': 'none'}
                    ),
                ],
                id='toolbar', 
                style={'display':'flex', 'flexGrow': '1', 'marginLeft': '20px', 'marginRight': '20px'}
            ),

            dcc.Store(id='open_silhouette_state', storage_type='memory'),
            html.Button('Silhouette Analysis', id='open_silhouette', n_clicks_timestamp=0, style={'display': 'none'}),

            dcc.Store(id='open_dendrogram_state', storage_type='memory'),
            html.Button('Dendrogram', id='open_dendrogram', n_clicks_timestamp=0, style={'display': 'none'}),

            dcc.Store(id='open_interpretation_state', storage_type='memory'),
            html.Button('Interpretation', id='open_interpretation', n_clicks_timestamp=0, style={'margin': '20px'}),

            
        ], style={'width': '100%', 'height':'12vh', 'display': 'flex', 'align-items': 'center'}),

        html.Hr(style={'margin': '0'}),

       
        dcc.Graph(
            id='PCA_graph',
            style={'height': '83vh', 'width': '100%'},
        ),

        html.Div(id='silhouette_overlay', children=[
            html.Div(children=[
                dcc.Graph(id='dimred_silhouette_scores')
            ], style={'width': '48%'}),

            html.Div(children=[
                dcc.Loading(children=[
                    dcc.Graph(id='dimred_silhouette')
                ])
            ], style={'width': '48%'})
        ], style={'diplay': 'none'}),

        html.Div(id='dendrogram_overlay', children=[
            html.Div(children=[
                dcc.Graph(id='dendrogram', style={'height': '83vh'})
            ], style={'width': '80%', 'height': '100%'}),
        ], style={'diplay': 'none'}),

        html.Div(id='interpretation_overlay', children=[
            html.Div(children=[
                
                dcc.Graph(id='interp_heatmap', style={'height': '83vh'}),
            ], style={'width': '35%', 'height': '100%', 'borderRight': '1px lightgray solid'}),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(id='interp_graph', style={'height': '53vh'}),
                    dcc.Graph(id='interp_linear', style={'height': '30vh'})
                ])
            ], style={'width': '65%'}),
        ], style={'diplay': 'none'}),

    ], style={'height': '95vh', 'width': '100%'}),

    dcc.ConfirmDialog(
        id='pca_error',
        message='Error: PCA requires at least 2 dimensions. Please select more features.',
    )
]

def serve_layout():
    session_id = str(uuid.uuid4())
    layout.append(ds.serve_layout(session_id))
    return layout


@app.server.route('/dimred/download')
def download_csv():
    value = flask.request.args.get('value')
    filename = 'temp/{}.xlsx'.format(value) 

    path = os.path.join(flask_app.instance_path, filename).replace('instance', '')
  

    def generate():
        with open(path, 'rb') as f:
            yield from f

        os.remove(path)

    r = flask_app.response_class(generate(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    r.headers.set('Content-Disposition', 'attachment', filename='data.xlsx')
    return r

"""
    General Layout
"""



"""
    Toobar Layout
"""
@app.callback(Output('dimred_clusters', 'style'), [Input('coloring_dd', 'value')])
def update_dimred_clustersn(value):
    if value == 'KM' or value == 'HC':
        return {'width': '12vw'}
    return {'display': 'none'}

@app.callback(Output('dimred_show', 'style'), [Input('coloring_dd', 'value')])
def update_dimred_show(value):
    if value == 'KM' or value == 'HC':
        return {'marginLeft': '20px'}
    return {'display': 'none'}

@app.callback(Output('open_silhouette', 'n_clicks_timestamp'), [Input('coloring_dd', 'value')], [State('open_silhouette', 'n_clicks_timestamp')])
def update_open_silhouette_timestamp(coloring, old):
    if coloring!='KM':
        return 0
    return old

@app.callback(Output('open_silhouette_state', 'data'), [Input('open_silhouette', 'n_clicks_timestamp')], [State('coloring_dd', 'value'), State('open_silhouette_state', 'data')])
def update_open_silhouette_state(time, coloring, old):
    if time==0 or coloring!='KM':
        return False

    return not old

@app.callback(Output('open_silhouette', 'style'), [Input('coloring_dd', 'value'), Input('open_silhouette_state', 'data')])
def update_open_silhouette(value, state):
    if value == 'KM':
        if state:
            return {'background-color':'black', 'color':'white', 'margin': '20px'}
        else:
            return {'margin': '20px'}
    return {'display': 'none'}

@app.callback(Output('silhouette_overlay', 'style'), [Input('open_silhouette_state', 'data')])
def update_silhouette_overlay(data):
    if data:
        return {'marginTop': '1px', 'height': '83vh', 'width': '100%', 'position':'absolute', 'top':'17vh', 'display': 'flex', 'alignItems': 'center', 'justifyContent':'center', 'backgroundColor': 'white'}
    return {'display': 'none'}

@app.callback(Output('open_dendrogram', 'n_clicks_timestamp'), [Input('coloring_dd', 'value')], [State('open_dendrogram', 'n_clicks_timestamp')])
def update_open_dendrogram_timestamp(coloring, old):
    if coloring!='HC':
        return 0
    return old

@app.callback(Output('open_dendrogram_state', 'data'), [Input('open_dendrogram', 'n_clicks_timestamp')], [State('coloring_dd', 'value'), State('open_dendrogram_state', 'data')])
def update_open_dendrogram_state(time, coloring, old):
    if time==0 or coloring!='HC':
        return False

    return not old

@app.callback(Output('open_dendrogram', 'style'), [Input('coloring_dd', 'value'), Input('open_dendrogram_state', 'data')])
def update_open_dendrogram(value, state):
    if value == 'HC':
        if state:
            return {'background-color':'black', 'color':'white', 'margin': '20px'}
        else:
            return {'margin': '20px'}
    return {'display': 'none'}

@app.callback(Output('dendrogram_overlay', 'style'), [Input('open_dendrogram_state', 'data')])
def update_dendrogram_overlay(data):
    if data:
        return {'marginTop': '1px', 'height': '83vh', 'width': '100%', 'position':'absolute', 'top':'17vh', 'display': 'flex', 'alignItems': 'center', 'justifyContent':'center', 'backgroundColor': 'white'}
    return {'display': 'none'}

@app.callback(Output('open_interpretation_state', 'data'), [Input('open_interpretation', 'n_clicks_timestamp')], [State('open_interpretation_state', 'data')])
def update_open_interpretation_state(time, old):
    if time==0:
        return False

    return not old

@app.callback(Output('open_interpretation', 'style'), [Input('open_interpretation_state', 'data')])
def update_open_interpretation(state):
    if state:
        return {'background-color':'black', 'color':'white', 'margin': '20px'}
    else:
        return {'margin': '20px'}

@app.callback(Output('interpretation_overlay', 'style'), [Input('open_interpretation_state', 'data')])
def update_interpretation_overlay(data):
    if data:
        return {'marginTop': '1px', 'height': '83vh', 'width': '100%', 'position':'absolute', 'top':'17vh', 'display': 'flex', 'alignItems': 'center', 'justifyContent':'center', 'backgroundColor': 'white'}
    return {'display': 'none'}


"""
    PCA Data Management
"""
@app.callback(Output('pca_data', 'data'), [Input('dimred____close_selector', 'n_clicks'), Input('coloring_dd', 'value')], [State('pca_error', 'displayed'), State('dimred____all_selected_data', 'data')])
def update_pca_data(n_clicks, method, pca_error, all_selected_data):
    if all_selected_data is None or pca_error or all_selected_data=={}:
        return {}

    data = pd.DataFrame.from_dict(all_selected_data['data'])
    features = data.columns[1:]

    if len(features)<2:
        return {'error': 0}

    x = data.loc[:, features].values

    for i in range(len(x)):
        for j in range(len(x[i])):
            if type(x[i][j]) is str and '-' in x[i][j]:
                x[i][j] = x[i][j].split('-')[0].strip()
    
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    principalDf_dict = principalDf.to_dict()

    return {'data': principalDf_dict, 'components': pca.components_, 'features': features}

@app.callback(Output('pca_error', 'displayed'), [Input('pca_data', 'data')])
def displayPCAError(data):
    if data == {'error': 0}:
        return True
    return False

"""
    K-means
"""
@app.callback(Output('km_data', 'data'), [Input('dimred_clusters-slider', 'value'), Input('dimred____close_selector', 'n_clicks'), Input('coloring_dd', 'value')], [State('dimred____all_selected_data', 'data')])
def update_km_data(n_clusters, n_clicks, method, all_selected_data):
    if all_selected_data is None or all_selected_data=={} or method!='KM':
        return {}

    data = pd.DataFrame.from_dict(all_selected_data['data'])
    data = clean_columns(data)
    
  
    clusters = KMeans(n_clusters=n_clusters, random_state=14)
    clusters.fit(data)

    df = pd.DataFrame(data = clusters.labels_ , columns = ['label'])
    df_dict = df.to_dict()
    return {'data': df_dict}

@app.callback(Output('dimred_silhouette_scores', 'figure'), [Input('open_silhouette_state', 'data')], [State('dimred____all_selected_data', 'data')])
def update_silhouette_figure(show, all_selected_data):
    if not show or all_selected_data=={}:
        return go.Figure()
    
    data = pd.DataFrame.from_dict(all_selected_data['data'])
    data = clean_columns(data)

    scores = []
  
    K = range(2,8)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=14)
        cluster_labels = kmeanModel.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        scores.append(silhouette_avg)

    return go.Figure(
        data=[
            go.Bar(
                x=list(K),
                y=scores
            )
        ],
        layout = go.Layout(
            
            xaxis=dict(
                title='Clusters'
            ),
            yaxis=dict(
                title='Average Silhouette Score'
            ),
            showlegend=False,
            margin=go.layout.Margin(
                l=70,
                r=70,
                b=70,
                t=70,
            )
        )
    )

@app.callback(Output('dimred_silhouette', 'figure'), [Input('dimred_silhouette_scores', 'hoverData')], [State('open_silhouette_state', 'data'), State('dimred____all_selected_data', 'data')])
def update_silhouette2(hoverdata, show, all_selected_data):
    if not show or all_selected_data=={}:
        return go.Figure()

    data = []

    all_data = pd.DataFrame.from_dict(all_selected_data['data'])
    all_data = clean_columns(all_data)


    k = hoverdata['points'][0]['x']

    kmeanModel = KMeans(n_clusters=k, random_state=14)
    cluster_labels = kmeanModel.fit_predict(all_data)
    silhouette_avg = silhouette_score(all_data, cluster_labels)
    silhouette_values = silhouette_samples(all_data, cluster_labels)

    max_score = max(silhouette_values)

    y_lower = 10
    
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                 x=ith_cluster_silhouette_values,
                                 mode='lines',
                                 showlegend=False,
                                 line=dict(width=0.5,
                                          color=colors[i]),
                                 fill='tozerox')
        data.append(filled_area)
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    axis_line = go.Scatter(x=[silhouette_avg],
                           y=[0, all_data.shape[0] + (k + 1) * 10],
                           showlegend=False,
                           mode='lines',
                           line=dict(color="red", dash='dash',
                                     width =1) )
    data.append(axis_line)

    return go.Figure(
        data = data,
        layout=go.Layout(
            titlefont=dict(
                size=11
            ),
            title="Hover over bars to the left to view analysis plot",
            xaxis=dict(
                title="Silhouette Coefficent",
                range=[-max_score, max_score],
            ),
            yaxis=dict(
                title="Cluster label",
                range=[0, all_data.shape[0] + (k + 1) * 10]
            ),
            margin=go.layout.Margin(
                l=70,
                r=50,
                b=30,
                t=30,
            )
        )
    )

"""
    Hierarchical Clustering
"""
@app.callback(Output('hc_data', 'data'), [Input('dimred_clusters-slider', 'value'), Input('dimred____close_selector', 'n_clicks'), Input('coloring_dd', 'value')], [State('dimred____all_selected_data', 'data')])
def update_hc_data(n_clusters, n_clicks, method, all_selected_data):
    if all_selected_data is None or all_selected_data=={} or method!='HC':
        return {}

    data = pd.DataFrame.from_dict(all_selected_data['data'])
    data = clean_columns(data)
    
    Z = linkage(data, 'ward')

    clusters = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')  
    clusters.fit_predict(data)  
    
    df = pd.DataFrame(data = clusters.labels_  , columns = ['label'])
    df_dict = df.to_dict()


    return {'data': df_dict, 'linkage': Z}

@app.callback(Output('dendrogram', 'figure'), [Input('hc_data', 'data')])
def update_dendrogram(hc_data):
    if hc_data == {}:
        return go.Figure()
    
    Z = np.array(hc_data['linkage'])

    figure = FF.create_dendrogram(
        Z, orientation='bottom',
        linkagefun=lambda x: linkage(Z, 'ward', metric='euclidean')
    )

    return figure

"""
    Main Graph
"""

@app.callback(Output('PCA_graph', 'figure'), 
    [Input('pca_data', 'data'), Input('km_data', 'data'), Input('hc_data', 'data'), Input('show_cl', 'value')],
    [State('coloring_dd', 'value'), State('dimred____all_selected_data', 'data')]
)
def update_PCA_figure(pca_data, km_data, hc_data, show_values, coloring, all_data):
    if pca_data=={} or pca_data=={'error': 0}:
        return go.Figure()

    pca_df = pd.DataFrame.from_dict(pca_data['data'])
    all_df = pd.DataFrame.from_dict(all_data['data'])
    
    all_df['x'] = pca_df['principal component 1']
    all_df['y'] = pca_df['principal component 2']

    data = []
    shapes = []

    
    show_centroids = 'Centroids' in show_values
    show_shapes = 'Cluster Shapes' in show_values

    if(coloring=='KM' or coloring=='HC'):
        temp = km_data if coloring=='KM' else hc_data

        temp_df = pd.DataFrame.from_dict(temp['data'])
        all_df['label'] = temp_df['label']

        for index, cluster in enumerate(sorted(all_df['label'].unique())):
            unique = all_df[all_df['label']==cluster]

            x = list(unique['x'])
            y = list(unique['y'])

            # custom_data = []

            # for a, b, c in zip(unique['original'], unique['justification'], unique['sentiment']):
            #     custom_data.append({'original': a, 'processed': b, 'sentiment': c})

            data.append(
                go.Scatter(
                    name='Cluster ' + str(cluster) + ' (n=' + str(len(x)) + ')',
                    x=x,
                    y=y,
                    # customdata=custom_data,
                    mode = 'markers',
                    marker = dict(
                        size = 6,
                        color = colors[index],
                    )
                )
            )
            data.append(
                go.Scatter(
                    name="Centroid",
                    x=[ sum(x) / float(len(x)) ],
                    y=[ sum(y) / float(len(y)) ],
                    mode = 'markers',
                    marker = dict(
                        size = 15,
                        color = colors[index],
                        opacity = 1.0 if show_centroids else 0.0,
                        line = dict(
                            width = 4,
                        )
                    )
                )
            )

            if show_shapes:
                shapes.append(
                    {
                        'type': 'circle',
                        'xref': 'x',
                        'yref': 'y',
                        'x0': min(x),
                        'y0': min(y),
                        'x1': max(x),
                        'y1': max(y),
                        'opacity': 0.2,
                        'fillcolor': colors[index],
                        'line': {
                            'color': colors[index],
                        },
                    }
                )
    else: 
        return go.Figure(
            data=[
                go.Scatter(
                    x=all_df['x'],
                    y=all_df['y'],
                    mode='markers'
                )
            ],
            layout=go.Layout(
                title='2 Component PCA',
                hovermode='closest',
                xaxis=dict(
                    title='PC1',
                    autorange=True,
                    showgrid=True,
                    zeroline=False,
                    showline=False,
                    ticks='',
                    showticklabels=False
                ),
                yaxis=dict(
                    title='PC2',
                    autorange=True,
                    showgrid=True,
                    zeroline=False,
                    showline=False,
                    ticks='',
                    showticklabels=False
                )
            )
        )

    return go.Figure(
                data=data,
                layout=go.Layout(
                    title='2 Component PCA',
                    legend={'orientation': 'h'},
                    margin=go.layout.Margin(
                        l=10,
                        r=10,
                        b=10,
                        t=100,
                    ),
                    hovermode='closest',
                    xaxis=dict(
                        autorange=True,
                        showgrid=True,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        showticklabels=False
                    ),
                    yaxis=dict(
                        autorange=True,
                        showgrid=True,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        showticklabels=False
                    ),
                    shapes=shapes,
                )
            )

"""
    Interpretation
"""
@app.callback(Output('interp_graph', 'figure'), [Input('open_interpretation_state', 'data'), Input('pca_data', 'data')])
def update_interpretation_graph(state, data):
    if not state or data=={}:
        return go.Figure()
    
    components = data['components']
    features = [word_wrap(x, 40) for x in data['features']]
    

    return go.Figure(
        data=[go.Scatter(
            x = components[0:][0],
            y = components[1:][0],
            mode = 'markers+text',
            text=features,
           
        

        )],
        layout=go.Layout(
            title='Principal Component 1 Score vs Principal Component 2 Score',
             hovermode='closest',
            yaxis=dict(
                title='Component 2 Score',
                rangemode='tozero',
                automargin=True
            ),
            xaxis=dict(
                title='Component 1 Score',
                rangemode='tozero',
                automargin=True
            )
        )
    )

@app.callback(Output('interp_heatmap', 'figure'), [Input('open_interpretation_state', 'data'), Input('pca_data', 'data')])
def update_interpretation_heatmap(state, data):
    if not state or data=={}:
        return go.Figure()
    
    components = data['components']

    transposed_components = [*zip(*components)]
    features = data['features']

    text = [['Score: {}<br>Feature: {}'.format(y, word_wrap(features[i], 25)) for y in x] for i, x in enumerate(transposed_components)]

    for i, feature in enumerate(features):
        if re.search(r'^Q\d{1,3}:', feature):
            features[i] = 'Survey ' + feature.split(':')[0]
        elif 'DP_' in feature:
            features[i] = feature.split(':')[0]

    return go.Figure(
        data=[go.Heatmap(
            z=transposed_components,
            x=['Component 1 Score', 'Component 2 Score'],
            y=features,
            colorscale = 'Viridis',
            # colorbar=dict(x=-0.3),
            hoverinfo='text',
            text=text

        )],
        layout=go.Layout(
            title='Principal Component Scores by Variable (Heatmap)',
            yaxis=dict(
                automargin=True,
                # side='right'
            ),
            xaxis=dict(
                automargin=True
            )
        )
    )

@app.callback(Output('interp_linear', 'figure'), [Input('open_interpretation_state', 'data'), Input('pca_data', 'data')])
def update_interpretation_linear(state, data):
    if not state or data=={}:
        return go.Figure()
    
    components = data['components']
    features = data['features']

    trace1 = go.Scatter(
        x=components[0:][0],
        y=[0]*len(features),
        mode='markers',
        hoverinfo='text',
        name='PCA Component 1',
        text=features
    )

    trace2 = go.Scatter(
        x=components[1:][0],
        y=[0]*len(features),
        mode='markers',
        hoverinfo='text',
        name='PCA Component 2',
        text=features
    )

    fig = tools.make_subplots(rows=2, cols=1, print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig['layout'].update(
        title='Principal Component Scores by Variable (Line)',
        showlegend=False,
        yaxis=dict(
            showgrid=False,
            ticks='',
            showticklabels=False
        ),
        yaxis2=dict(
            showgrid=False,
            ticks='',
            showticklabels=False
        ),
        annotations=[
            dict(
                x=0,
                y=0.75,
                showarrow=False,
                text='Component 1 Score',
                xref='paper',
                yref='paper'
            ),
            dict(
                x=0,
                y=0,
                showarrow=False,
                text='Component 2 Score',
                xref='paper',
                yref='paper'
            ),
        ]
    
    )

    return fig