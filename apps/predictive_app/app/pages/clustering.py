from app import dash_app as app
from app import app as flask_app
from app import tab_selected_style, tab_style

import dash

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotly import tools

from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
from sklearn.cluster import KMeans

from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from nltk import tokenize


from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score

from canvasModels import *
import canvasModels
from app_predictive import tab_style, tab_selected_style
from app_data_manager import assignment_names, assignment_to_ratingIDs

import time
import string
import os




"""
    GENERAL SETUP
"""
debug_path = os.path.join(flask_app.instance_path, 'DEBUG.txt').replace('instance', '')
DEV = os.path.isfile(debug_path)

if not DEV:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

db = canvasModels.init_db(sqlCredentials)

stop_words = stopwords.words('english')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



def calcParagraphSentiment(x):
    sentence_list = tokenize.sent_tokenize(x)

    if(len(sentence_list)==0):
        return 0.0 

    paragraphSentiments = 0.0
    for sentence in sentence_list:
        # print("{:-<69} {}".format(sentence, str(vs["compound"])))
        paragraphSentiments += sid.polarity_scores(sentence)['compound']
    return paragraphSentiments / len(sentence_list)

def lemmatize(x):
    result = ''

    nltk_tokens = nltk.word_tokenize(x)
    #Next find the roots of the word
    for w in nltk_tokens:
        result += wordnet_lemmatizer.lemmatize(w) + ' '

    return result

"""
    LAYOUT
"""
default_tabs = [
    dcc.Tab(label='Frequency', value='frequency', children=[
        html.Div(children=[
            dcc.Graph(
                id='word_freq',
                style={'height': '38vh'}
            ),

            dcc.Checklist(
                    id='co-occurence',
                    options=[
                        {'label': 'Co-occurence', 'value': 'show'},
                    ],
                    value=[],
                    style={'position': 'absolute', 'top': '1rem', 'right': '1rem'}
            )
        ], style={'position': 'relative'}),
            
        html.Div(children=[
            html.Div(children=[
                html.H6('Custom Stopwords', style={'marginTop': '1rem', 'marginBottom': '0rem'}),
                dcc.Dropdown(
                    id='custom_stopwords',
                    options=[],
                    value=[],
                    multi=True,
                ),
                html.P('Click on bars above to add.'),
            ], style={'width': '50%', 'marginRight': '3rem'}),

            html.Div(children=[
                html.H6('Frequency Filter', style={'marginTop': '1rem', 'marginBottom': '0rem'}),
                dcc.Slider(
                    min=0,
                    max=10,
                    step=0.1,
                    value=0,
                    id='diff-slider',
                    className='margin-1rem ', 
                ),
                html.P("Off", style={'marginTop': '1rem'}, id="diff-desc")

            ], style={'width': '50%'}),
        ], style={'display': 'flex'}),        
    ], style=tab_style, selected_style=tab_selected_style),
    dcc.Tab(label='Sentiment', value='sentiment', children=[
        dcc.Graph(
            id='sentiment',
            style={'height': '40vh'}
        ),
        html.P("positive sentiment: compound score >= 0.05"),
        html.P("neutral sentiment: -0.05 < compound score < 0.05"),
        html.P("negative sentiment: compound score <= -0.05")
    ], style=tab_style, selected_style=tab_selected_style),
    dcc.Tab(label='Text', value='text', children=[
        dcc.Textarea(
            id='justification',
            placeholder='Hover over data point to read text.',
            value='',
            style={'width': '100%', 'height': '40vh', 'marginTop': '1rem'},
            readOnly=True,
        ),
    ], style=tab_style, selected_style=tab_selected_style),
]

layout = [
    dcc.Store(id='all_justifications', storage_type='memory'),

    html.Div(children=[
        dcc.Graph(
            id='tSNE_graph',
            style={'height': '100%', 'width': '70vw'},
        ),

        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                    html.H5('Text', style={'marginTop': '1rem', 'marginBottom': '0rem'}),
                    dcc.Dropdown(
                        options=[{'label': x, 'value': x} for x in ['Innovation Ratings', 'Decision Point']],
                        id='text_selector',
                        style={'marginRight': '2rem'}
                    ),
                ], style={'width': '50%'}),
                html.Div(children=[
                    html.H5('Assignment', style={'marginTop': '1rem', 'marginBottom': '0rem'}),
                    dcc.Dropdown(

                        id='clustering_assignment_selector',
                    ),
                ], style={'width': '50%'}),
            ], style={'display': 'flex'}),
            
            html.Div(children=[
                dcc.RadioItems(
                    options=[
                        {'label': x, 'value': x} for x in ['Technology', 'Market', 'Organization']
                    ],
                    labelStyle={'display': 'inline-block', 'padding-right': '5px'},
                    style={'display': 'none'},
                    id='rating_type'
                ),
            ]),

            html.Div(children=[
                html.Div(children=[
                    html.H5('Clusters', style={'marginBottom': '0rem'}),
                    dcc.RadioItems(
                        options=[
                            {'label': 'Decision Point', 'value': 'Decision Point'},
                            {'label': 'K Means', 'value': 'K Means'},
                        ],
                        value='Decision Point',
                        id='coloring'
                    ),
                ], style={'width': '33%'}),

                html.Div(children=[
                    html.H5('Show:', style={'marginBottom': '0rem'}),
                    dcc.Checklist(
                        id='show',
                        options=[
                            {'label': 'Centroids', 'value': 'Centroids'},
                            {'label': 'Cluster Shapes', 'value': 'Cluster Shapes'}
                        ],
                        value=['Centroids']
                    )
                ], style={'width': '33%'}),

                html.Div(children=[
                    html.H5('Clusters: 2', style={'marginBottom': '0rem'}, id='clusters-label'),
                    dcc.Slider(
                        min=2,
                        max=5,
                        step=1,
                        value=2,
                        id='clusters-slider',
                        className='margin-1rem ', 
                    )
                ], style={'display': 'none', 'width': '33%'}, id='clusters-container')
            ], style={'display': 'flex', 'marginTop': '1rem'}),

            dcc.Tabs(id="tabs", value='frequency', children=default_tabs, style={ 'marginTop': '1rem'}),
            html.Hr(style={'margin': '10'}),
            dcc.Graph(
                id='silhouette',
                style={'height': '25vh'},
                figure=go.Figure(
                    
                    layout=go.Layout(
                        titlefont=dict(
                            size=11
                        ),
                        title="Hover over bars above to view analysis plot",
                    )
                )
            )
        ], style={'width': '30vw'})
    ], style={'height': '92vh', 'display': 'flex', 'marginRight': '1rem'}),
]

def get_dp_justifications(assignment_name):
    pre_post = 'pre'

    optionA = None
    optionB = None

    all_justifications = pd.DataFrame(columns=['student', 'decision', 'justification'])
    for assignment in assignment_to_ratingIDs[assignment_name][pre_post]:
        dps = (DecisionPoints.select()
            .where(DecisionPoints.assignment==assignment)
            .where(DecisionPoints.pre_or_post==pre_post)
        )
        for dp in dps:
            if 'Option A' in dp.decision:
                if(optionA is None):
                    optionA = dp.decision

            elif 'Option B' in dp.decision:
                if(optionB is None):
                    optionB = dp.decision
            else:
                continue
                
            all_justifications = all_justifications.append({'student': dp.student_id, 'decision': dp.decision, 'justification': dp.justification}, ignore_index=True)

    optionA = optionA.split(' - ')[1]
    optionB = optionB.split(' - ')[1]

    custom_stopwords = [assignment_name.lower()]
    for option in [optionA, optionB]:
        for word in lemmatize(option).split():
            custom_stopwords.append(word)

    for option in [optionA, optionB]:
        option = option.replace('-', ' ')

        for x in option.split():
            if x.lower() not in custom_stopwords:
                custom_stopwords.append(x.lower())


    temp_qr = QuizResponses.select().where(QuizResponses.assignment==assignment).where(QuizResponses.question.contains('Decision Point')).first()
    question = temp_qr.question.split('Decision Point: ')[1]

    for word in lemmatize(question).split():
        custom_stopwords.append(word.lower())

    return all_justifications, question, custom_stopwords

def get_ratings_justifications(assignment_name, pre_post, rating_type):
    all_justifications = pd.DataFrame(columns=['student', 'decision', 'justification'])

    for assignment in assignment_to_ratingIDs[assignment_name][pre_post]:
        ratings = (InnovationRatings.select()
            .where(InnovationRatings.assignment==assignment.assignment_id)
            .where(InnovationRatings.rating_type==rating_type))
        
        for rating in ratings:
            all_justifications = all_justifications.append({'student': rating.student_id, 'decision': rating.rating, 'justification': rating.justification}, ignore_index=True)
    
    return all_justifications, '{} ratings for {} ({})'.format(rating_type, assignment_name, pre_post), [rating_type.lower(), assignment_name.lower()]

"""
    GENERAL LAYOUT/OTHER
"""
@app.callback(Output('coloring', 'options'), [Input('text_selector', 'value')])
def update_coloring_options(value):
    if(value=='Innovation Ratings'):
        return [
            {'label': 'Rating', 'value': 'Decision Point'},
            {'label': 'K Means', 'value': 'K Means'},
        ]
    
    return [
        {'label': 'Decision Point', 'value': 'Decision Point'},
        {'label': 'K Means', 'value': 'K Means'},
    ]

@app.callback(Output('rating_type', 'style'), [Input('text_selector', 'value'), Input('clustering_assignment_selector', 'value')])
def hide_rating_type(value, assignment):
    if(value=='Innovation Ratings' and assignment != None):
        return {'textAlign': 'center', 'marginTop':'1rem'}
    
    return {'display': 'none'}

@app.callback(Output('clustering_assignment_selector', 'disabled'), [Input('text_selector', 'value')])
def disable_clustering_assignment_selector(value):
    if(value is None):
        return True
    return False

@app.callback(Output('clustering_assignment_selector', 'options'), [Input('text_selector', 'value')])
def update_assignments(text):
    if(text is None):
        return []
    elif(text=='Innovation Ratings'):
        options = []
        for x in assignment_names[0:-3]:
            for y in [' pre-discussion', ' post-discussion']:
                options.append(x + y)

        return [{'label': x, 'value': x} for x in options]
    elif(text=='Decision Point'):
        return [{'label': x, 'value': x} for x in assignment_names[1:-3]]

@app.callback(Output('clusters-container', 'style'), [Input('coloring', 'value')])
def hide_clusters_container(coloring):
    if(coloring=='K Means'):
        return {'width': '33%'}
    else:
        return {'display': 'none', 'width': '33%'}

@app.callback(Output('tabs', 'children'), [Input('coloring', 'value')])
def update_tabs(value):
    if(value=='K Means'):
        return default_tabs + [
            dcc.Tab(label='Best K', value='bestK', children=[
                dcc.Graph(
                    id='silhouette_scores',
                    style={'height': '25vh'}
                ),
                
            ], style=tab_style, selected_style=tab_selected_style)
        ]
    
    return default_tabs

@app.callback(Output('silhouette', 'style'), [Input('tabs', 'value')])
def hide_silhouette_graph(tab):
    if tab == 'bestK':
        return {'height': '25vh'}
    return {'display': 'none'}

@app.callback(Output('clusters-label', 'children'), [Input('clusters-slider', 'value')])
def update_clusters_label(value):
    return 'Clusters: ' + str(value)


"""
    DATA Management
"""
@app.callback(Output('all_justifications', 'data'), [Input('text_selector', 'value'), Input('clustering_assignment_selector', 'value'), Input('rating_type', 'value')])
def update_justifications_data(text, assignment_name, rating_type):
    if(assignment_name is None):
        return None
    
    global db

    db.close()
    db.connect()

    if(text=='Decision Point'):
        all_justifications, title, custom_stopwords = get_dp_justifications(assignment_name)
    elif(text=='Innovation Ratings'):
        if(rating_type is None):
            return None

        assignment = assignment_name.split()[0]
        pre_post = assignment_name.split()[1].split('-')[0]
        all_justifications, title, custom_stopwords = get_ratings_justifications(assignment, pre_post, rating_type)

    custom_stopwords.extend(['wa', 'could', 'also', 'would', 'ha', 'i', 'p', 'g', 'this', 'the'])

    all_justifications = all_justifications[ [True if len(x)>2 else False for x in all_justifications['justification']] ]
    all_justifications['justification'].replace('', np.nan, inplace=True)
    all_justifications.dropna(inplace=True)
    all_justifications['original'] = all_justifications['justification']
    all_justifications['justification'] = all_justifications['justification'].apply(lambda x: lemmatize(x))
    all_justifications['sentiment'] = all_justifications['justification'].apply(lambda x: calcParagraphSentiment(x))

    tfidf = TfidfVectorizer(stop_words=custom_stopwords)
    docs = tfidf.fit_transform(list(all_justifications['justification']))

    tsne = TSNEVisualizer(random_state=14)
    transformer = tsne.make_transformer()
    data = transformer.fit_transform(docs)

    all_justifications['x'] = data[:, 0]
    all_justifications['y'] = data[:, 1]

    return {'data': all_justifications.to_dict(), 'title': title, 'custom_stopwords': custom_stopwords}

"""
    FREQUENCY Tab: displays bar graphs of most common words per cluster
"""
@app.callback(Output('word_freq', 'figure'), [Input('tSNE_graph', 'figure'), Input('custom_stopwords', 'value'), Input('diff-slider', 'value'), Input('co-occurence', 'value')], [State('all_justifications', 'data')])
def update_histo(figure, extra_stopwords, diff, show_occurence_values, raw):
    bars = 10 if 'show' in show_occurence_values else 20
    tickangle = 20 if 'show' in show_occurence_values else 35
    margin = 80 if 'show' in show_occurence_values else 50

    if(len(figure['data'])==0):
        return go.Figure()

    fig = tools.make_subplots(rows=int(len(figure['data'])/2), cols=1, print_grid=False)

    custom_stopwords = raw['custom_stopwords']
    custom_stopwords.extend(extra_stopwords)

    counters = []
    names = []
    intersect = set()

    for index, justifications in enumerate(figure['data']):
        tokens = []

        if 'customdata' not in justifications:
            continue

        names.append(justifications['name'] if 'Option ' not in justifications['name'] else justifications['name'].split(' - ')[0])
        
        for justification in justifications['customdata']:
            if('show' in show_occurence_values):
                split = justification['original'].translate(str.maketrans('','',string.punctuation)).split()
                for i in range(0, len(split)-1):
                    if(not split[i].isalpha() or not split[i+1].isalpha() ):
                        continue
                    
                    if(split[i].lower() in stop_words or split[i+1].lower() in stop_words ):
                        continue

                    if(split[i].lower() in custom_stopwords or split[i+1].lower() in custom_stopwords ):
                        continue
                    
                    key = lemmatize(split[i].lower()) + lemmatize(split[i+1].lower())
                    
                    if(key not in custom_stopwords):
                        tokens.append(key)
            else:
                tokens.extend([x.lower() for x in justification['processed'].split() if x not in stop_words and x.isalpha() and x.lower() not in custom_stopwords])

        fdist = nltk.FreqDist(tokens)

        # print(tokens)

        counters.append(fdist)

        if(diff!=0):
            if(index==0):
                intersect = fdist.keys()
            else:
                intersect = intersect & fdist.keys()

    for key in intersect:
        counts = []

        for counter in counters:
            counts.append(counter[key] / float(sum(counter.values())))

        if(max(counts) - min(counts) < diff/100.0):
            for counter in counters:
                if key in counter.keys():
                    counter.pop(key)

    for index, fdist in enumerate(counters):
        x = []
        y = []
        for word, freq in fdist.most_common(bars):
            x.append(word)
            y.append(freq)

        fig.append_trace(go.Bar(
            name=names[index],
            x = x,
            y = y,
            text = [names[index] + "<br />" + str(a) + " words / " + str(round(a / float(len(x)), 2 )) + " per student" for a in y],
            hoverinfo = 'text',
        ), index+1, 1)
    
    fig['layout'].update(margin=dict(l=0,r=margin,b=margin,t=0))
    fig['layout'].update(showlegend=False)
    fig['layout'].update(xaxis=dict(tickangle=tickangle))
    fig['layout'].update(xaxis2=dict(tickangle=tickangle))
    fig['layout'].update(xaxis3=dict(tickangle=tickangle))
    fig['layout'].update(xaxis4=dict(tickangle=tickangle))
    fig['layout'].update(xaxis5=dict(tickangle=tickangle))
    
    
    return fig

@app.callback(Output('custom_stopwords', 'options'), [Input('word_freq', 'clickData')], [State('custom_stopwords', 'value')])
def update_dropdown(clickdata, old_options):
    if(clickdata is None):
        return []

    old_options = [{'label': x, 'value': x} for x in old_options]

    word = clickdata['points'][0]['x']
    option = {'label': word, 'value': word}

    if option not in old_options:
        old_options.append(option)
    
    return old_options

@app.callback(Output('custom_stopwords', 'value'), [Input('custom_stopwords', 'options')])
def update_value(options):
    if options is None:
        return []
    return [x['value'] for x in options]

@app.callback(Output("diff-desc", "children"), [Input("diff-slider", "value")])
def updateDiffDescription(value):
    if(value==0):
        return "Off"

    return "Only show words whose frequencies differ by " + str(value) + "%"

"""
    SENTIMENT Tab
"""
@app.callback(Output('sentiment', 'figure'), [Input('tSNE_graph', 'figure')], [State('all_justifications', 'data')])
def update_sentiment(figure, raw):
    if(len(figure['data'])==0):
        return go.Figure()
    
    data = []

    for index, justifications in enumerate(figure['data']):
        if 'customdata' not in justifications:
            continue

        name = justifications['name']

        sentiments = [x['sentiment'] for x in justifications['customdata']]
        average =  sum(sentiments) / float(len(sentiments)) 

        data.append(
            go.Bar(
                x=[average],
                y=[name],
                orientation = 'h'
            )
        )



    # data.reverse()
    return go.Figure(
        data=data,
        layout = go.Layout(
            xaxis=dict(
                range=[-1, 1]
            ),
            yaxis=dict(
                showticklabels=False
            ),
            showlegend=False,
            margin=go.layout.Margin(
                l=50,
                r=50,
                b=50,
                t=50,
            )
        )
    )

"""
    TEXT Tab
"""
@app.callback(Output('justification', 'value'), [Input('tSNE_graph', 'hoverData')])
def update_text(hoverdata):
    if(hoverdata is None):
        return ''

    return hoverdata['points'][0]['customdata']['original']

@app.callback(Output('tSNE_graph', 'figure'), [Input('all_justifications', 'data'), Input('coloring', 'value'), Input('clusters-slider', 'value'), Input('show', 'value')])
def update_tSNE_graph(raw, coloring, n_clusters, show_values):
    if(raw is None):
        return go.Figure()

    all_justifications = pd.DataFrame(raw['data'])

    tfidf = TfidfVectorizer()
    docs = tfidf.fit_transform(list(all_justifications['justification']))

    data = []
    shapes = []

    show_centroids = 'Centroids' in show_values

    if(coloring=='Decision Point'):
        for index, decision in enumerate(sorted(all_justifications['decision'].unique())):
            unique = all_justifications[all_justifications['decision']==decision]

            x = list(unique['x'])
            y = list(unique['y'])

            custom_data = []

            for a, b, c in zip(unique['original'], unique['justification'], unique['sentiment']):
                custom_data.append({'original': a, 'processed': b, 'sentiment': c})

            if(len(decision) > 140):
                decision = decision[0:140] + "..."
    
 

            data.append(
                go.Scatter(
                    name=decision + ' (n=' + str(len(x)) + ')',
                    x=x,
                    y=y,
                    customdata=custom_data,
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
                        opacity = 1.0 if show_centroids else 0.0,
                        color = colors[index],
                        line = dict(
                            width = 4,
                        )
                    )
                )
            )
            if('Cluster Shapes' in show_values):
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
    elif(coloring=='K Means'):
        clusters = KMeans(n_clusters=n_clusters, random_state=14)
        clusters.fit(docs)

        all_justifications['kmeans'] = clusters.labels_

        for index, decision in enumerate(sorted(all_justifications['kmeans'].unique())):
            unique = all_justifications[all_justifications['kmeans']==decision]

            x = list(unique['x'])
            y = list(unique['y'])

            custom_data = []

            for a, b, c in zip(unique['original'], unique['justification'], unique['sentiment']):
                custom_data.append({'original': a, 'processed': b, 'sentiment': c})

            data.append(
                go.Scatter(
                    name='Cluster ' + str(decision) + ' (n=' + str(len(x)) + ')',
                    x=x,
                    y=y,
                    customdata=custom_data,
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

            if('Cluster Shapes' in show_values):
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

    return go.Figure(
                data=data,
                layout=go.Layout(
                    title=raw['title'],
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
    BEST K Tab
"""
@app.callback(Output('silhouette_scores', 'figure'), [Input('all_justifications', 'data'), Input('tabs', 'children'), Input('tabs', 'value')])
def updateKGraph(raw, throwaway, tab):
    if(tab!='bestK'):
        return

    all_justifications = pd.DataFrame(raw['data'])

    tfidf = TfidfVectorizer()
    docs = tfidf.fit_transform(list(all_justifications['justification']))

    scores = []
  
    K = range(2,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=14)
        cluster_labels = kmeanModel.fit_predict(docs)
        silhouette_avg = silhouette_score(docs, cluster_labels)
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
                r=50,
                b=30,
                t=20,
            )
        )
    )

@app.callback(Output('silhouette', 'figure'), [Input('silhouette_scores', 'hoverData')], [State('all_justifications', 'data')])
def updateSilhouetteFigure(hoverdata, raw):
    data = []

    all_justifications = pd.DataFrame(raw['data'])

    tfidf = TfidfVectorizer()
    docs = tfidf.fit_transform(list(all_justifications['justification']))

    k = hoverdata['points'][0]['x']

    kmeanModel = KMeans(n_clusters=k, random_state=14)
    cluster_labels = kmeanModel.fit_predict(docs)
    silhouette_avg = silhouette_score(docs, cluster_labels)
    silhouette_values = silhouette_samples(docs, cluster_labels)

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
                           y=[0, docs.shape[0] + (k + 1) * 10],
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
            title="Hover over bars above to view analysis plot",
            xaxis=dict(
                title="Silhouette Coefficent",
                range=[-max_score, max_score],
            ),
            yaxis=dict(
                title="Cluster label",
                range=[0, docs.shape[0] + (k + 1) * 10]
            ),
            margin=go.layout.Margin(
                l=70,
                r=50,
                b=30,
                t=30,
            )
        )
    )

