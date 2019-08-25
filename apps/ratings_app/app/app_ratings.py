import peewee
from peewee import *

import canvasModels
from canvasModels import *
from canvasapi import Canvas
import sqlCredentials
import canvasCredentials

import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from app import dash_app as app

import random
import math
import time

import requests
import json
import pandas as pd
from pandas.compat import StringIO
import datetime
import numpy as np

db = canvasModels.init_db(sqlCredentials)

colors = ['#e5e523', '#1f9274', '#36226a']
canvas = Canvas(canvasCredentials.API_URL, canvasCredentials.API_KEY)

def graph(id, y, title):
    return dcc.Graph(
        id=id,
        figure={
            'data': [
                {'x': ['Low', 'Medium', 'High'], 'y': y, 'type': 'bar', 'marker': dict(color=colors)},
            ],
            
            'layout': {
                'title': title,
                'margin': {'t': 50, 'b': 50},
                'yaxis': {'range': [0, 50]},
             
            },
            
        },
        style={'height':'100%', 'width':'100%', 'padding':'0'}
    )

def delta_graph(id, y, title):
    return dcc.Graph(
        id=id,
        figure={
            'data': [
                {'x': ['Low', 'Medium', 'High'], 'y': y, 'type': 'bar', 'marker': dict(color=colors)},
            ],
            
            'layout': {
                'margin': {'t': 25, 'b': 25},
                'yaxis': {'range': [-5, 5]},
             
            },
            
        },
        style={'height':'100%', 'width':'100%', 'padding':'0'}
    )

def get_section_to_rating_assignments(sections):
    section_to_rating_assignment = dict()

    for section in sections:
        
        section_to_rating_assignment[section] = {}

        assignments = (Assignments
                .select()
                .join(Sections)
                .where(Assignments.section==section.canvas_id))
            
        for assignment in assignments:
            if "pre-discussion" in assignment.name or "post-discussion" in assignment.name:
                key = assignment.name.replace(" pre-discussion", "").replace(" post-discussion", "").strip()
                
                if key not in section_to_rating_assignment[section]:
                    section_to_rating_assignment[section][key] = [0, 0]

                if "pre-discussion" in assignment.name:
                    section_to_rating_assignment[section][key][0] = assignment
                elif "post-discussion" in assignment.name:
                    section_to_rating_assignment[section][key][1] = assignment

    return section_to_rating_assignment

sections = Sections.select()
sections_dropdown = [{'label': y.name, 'value': x} for x, y in enumerate(sections)]
sections_dropdown.append({'label': 'All Sections', 'value': len(sections_dropdown)})

section_to_rating_assignments = get_section_to_rating_assignments(sections)
cases = [list(x.keys()) for x in section_to_rating_assignments.values()] 

hidden_style = {"display": "none"}
hidden_inputs = html.Div(id="hidden-inputs", style=hidden_style, children=[])

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')

layout = html.Div([
    html.Div([
        html.Div([
             html.Div([
                html.Button('Counts', 
                    id='units_button',
                    style={'marginRight': '10px'}
                ),
                html.Button('All', 
                    id='data_button',
                ),
            ], style={'marginTop':'5vh', 'transform':'translateY(-50%)', 'marginLeft': '1rem'}),

            html.Div([
                dcc.Dropdown(
                    id='section-dropdown',
                    options=sections_dropdown,
                    value=0,
                ),
            ], style={'marginTop': 'auto', 'marginBottom':'auto'}),

            html.Div([
                dcc.Dropdown(
                    id='case-dropdown',
                    value=0,
                ),
            ], style={'marginTop': 'auto', 'marginBottom':'auto'}),

            html.Div([
                # html.Button('Refresh', 
                #     id='refresh_button',
                #     style={}
                # ),
            ], style={'marginTop':'5vh', 'transform':'translateY(-50%)', 'marginRight': '4rem', 'marginLeft': 'auto'}),


        ], className="gridHeader"),
    ], style={'position': 'relative', 'border-bottom': '1px solid lightgrey'}),
    
    
    html.Div([
        html.Div([
            html.H5(
                children="Pre-discussion Ratings",
            ),
            
        ], className="gridTitle"),

        html.Div([
            graph("Pre-Technology", [0, 0, 0], "Technology")
        ], style={'border-right': '1px solid lightgrey'}),

        html.Div([
            graph("Pre-Market", [0, 0, 0], "Market")
        ], style={'border-right': '1px solid lightgrey'}),

        html.Div([
            graph("Pre-Organization", [0, 0, 0], "Organization")
        ]),

        html.Div([
            html.H5(
                children="Post-discussion Ratings",
            ),
        ], className="gridTitle"),

        html.Div([
            graph("Post-Technology", [0, 0, 0], "Technology")
        ], style={'border-right': '1px solid lightgrey'}),

        html.Div([
            graph("Post-Market", [0, 0, 0], "Market")
        ], style={'border-right': '1px solid lightgrey'}),

        html.Div([
            graph("Post-Organization", [0, 0, 0], "Organization")
        ]),

        html.Div([
            html.H5(
                children="Change in Ratings",
            ),
        ], className="gridTitle"),

        html.Div([
            delta_graph("delta-Technology", [0, 0, 0], "Organization")
        ], style={'border-right': '1px solid lightgrey'}),
        html.Div([
            delta_graph("delta-Market", [0, 0, 0], "Organization")
        ], style={'border-right': '1px solid lightgrey'}),
        html.Div([
            delta_graph("delta-Organization", [0, 0, 0], "Organization")
        ]),

    ], className="gridContainer"),
    
       
    html.Div([
        html.Button('Close', 
            id='button',
            style={'position': 'absolute', 'bottom': '1rem', 'right': '1rem'}
        ),
        html.Div([
            dcc.Loading(children=[
                dt.DataTable(
                    row_selectable=False,
                    filtering=True,
                    sorting=True,
                    id='datatable',
                    style_table={'height': '25vh', 'overflowY': 'scroll'},
                    style_data={'whiteSpace': 'normal'},
                    css=[{
                        'selector': '.dash-cell div.dash-cell-value',
                        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                    }],
                ),
                html.Div('', id='justification_data', style={'display': 'none'}),
            ])
            
        ], style={'width':'90%', 'margin':'auto', 'top':'50%', 'transform':'translateY(-50%)', 'position':'relative'})
    ], className='detailedView2', id='tableView', style={'display': 'none'}),

    html.Div([
        html.Button('Close', 
            id='close_delta_button',
            style={'position': 'absolute', 'bottom': '1rem', 'right': '1rem'}
        ),
        

        dcc.Loading(children=[
            html.Div([
                dcc.Tabs(id="tabs", children=[
                    dcc.Tab(label='Switched To', children=[
                        html.Div([
                            
                                dt.DataTable(
                                    id='delta_datatable1',
                                    data=[{'Loading...': 'Retrieving Data...'}], # initialise the rows
                                    row_selectable=False,
                                    filtering=True,
                                    sorting=True,
                                    style_table={'height': '25vh', 'overflowY': 'scroll'},
                                    style_data={'whiteSpace': 'normal'},
                                    css=[{
                                        'selector': '.dash-cell div.dash-cell-value',
                                        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                    }],
                                ),
                                html.Div('', id='delta_data1', style={'display': 'none'}),
                        
                        ]),
                    ]),

                    dcc.Tab(label='Switched From', children=[
                        html.Div([
                        
                                dt.DataTable(
                                    id='delta_datatable2',
                                    data=[{'Loading...': 'Retrieving Data...'}], # initialise the rows
                                    row_selectable=False,
                                    filtering=True,
                                    sorting=True,
                                    style_table={'height': '25vh', 'overflowY': 'scroll'},
                                    style_data={'whiteSpace': 'normal'},
                                    css=[{
                                        'selector': '.dash-cell div.dash-cell-value',
                                        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                    }],
                                ),
                                html.Div('', id='delta_data2', style={'display': 'none'}),
                            
                        ]),
                    ])
                ])
            ], style={'width': '90%', 'margin': 'auto'}),
        ], style={'width':'90%', 'margin':'4rem', 'top':'50%', 'transform':'translateY(-50%)', 'position':'relative'})
    ], className='detailedView2', id='deltaTableView', style={'display':'none'}),

    html.Div('', id='selectedSection', style={'display':'none'}),
    html.Div('', id='ratings', style={'display':'none'}),
    html.Div('', id='ratings_students', style={'display':'none'}),
    html.Div('', id='previous_clickData', style={'display':'none'}),
    html.Div('', id='previous_delta_clickData', style={'display':'none'}),
    html.Div('0', id='refresh_count', style={'display':'none'}),
   
    
   
    hidden_inputs,
])

# last_clicked method from https://gist.github.com/shawkinsl/22a0f4e0bf519330b92b7e99b3cfee8a
def last_clicked(*dash_input_keys):
    """ Get the clickData of the most recently clicked graph in a list of graphs.
    The `value` you will receive as a parameter in your callback will be a dict. The keys you will want to
    pay attention to are:
        - "last_clicked": the id of the graph that was last clicked
        - "last_clicked_data": what clickData would usually return
    This function working depends on a `hidden_inputs` variable existing in the global / file scope. It should be an
    html.Div() input with styles applied to be hidden ({"display": "none"}).
    but why, I hear you ask?
    clickData does not get set back to None after you've used it. That means that if a callback needs the latest
    clickData from two different potential clickData sources, if it uses them both, it will get two sets of clickData
    and no indication which was the most recent.
    :type dash_input_keys: list of strings representing dash components
    :return: dash.dependencies.Input() to watch value of
    """
    dash_input_keys = sorted(list(dash_input_keys))
    str_repr = str(dash_input_keys)
    last_clicked_id = str_repr + "_last-clicked"
    existing_child = None
    for child in hidden_inputs.children:
        if child.id == str_repr:
            existing_child = child
            break

    if existing_child:
        return Input(last_clicked_id, 'value')

    # If we get to here, this is the first time calling this function with these inputs, so we need to do some setup
    # make feeder input/outputs that will store the last time a graph was clicked in addition to it's clickdata
    if existing_child is None:
        existing_child = html.Div(id=str_repr, children=[])
        hidden_inputs.children.append(existing_child)

    input_clicktime_trackers = [str_repr + key + "_clicktime" for key in dash_input_keys]
    existing_child.children.append(dcc.Input(id=last_clicked_id, style=hidden_style, value=None))
    for hidden_input_key in input_clicktime_trackers:
        existing_child.children.append(dcc.Input(id=hidden_input_key, style=hidden_style, value=None))

    # set up simple callbacks that just append the time of click to clickData
    for graph_key, clicktime_out_key in zip(dash_input_keys, input_clicktime_trackers):
        @app.callback(Output(clicktime_out_key, 'value'),
                      [Input(graph_key, 'clickData')],
                      [State(graph_key, 'id')])
        def update_clicktime(clickdata, graph_id):
            result = {
                "click_time": datetime.datetime.now().timestamp(),
                "click_data": clickdata,
                "id": graph_id
            }
            return result

    cb_output = Output(last_clicked_id, 'value')
    cb_inputs = [Input(clicktime_out_key, 'value') for clicktime_out_key in input_clicktime_trackers]
    cb_current_state = State(last_clicked_id, 'value')

    # use the outputs generated in the callbacks above _instead_ of clickData
    @app.callback(cb_output, cb_inputs, [cb_current_state])
    def last_clicked_callback(*inputs_and_state):
        clicktime_inputs = inputs_and_state[:-1]
        last_state = inputs_and_state[-1]
        if last_state is None:
            last_state = {
                "last_clicked": None,
                "last_clicked_data": None,
            }
        else:
            largest_clicktime = -1
            largest_clicktime_input = None
            for clicktime_input in clicktime_inputs:
                if clicktime_input is None:
                    continue
                click_time = int(clicktime_input['click_time'])
                if clicktime_input['click_data'] and click_time > largest_clicktime:
                    largest_clicktime_input = clicktime_input
                    largest_clicktime = click_time
            if largest_clicktime:
                last_state['last_clicked'] = largest_clicktime_input["id"]
                last_state['last_clicked_data'] = largest_clicktime_input["click_data"]
        return last_state

    return Input(last_clicked_id, 'value')

def get_rating_counts(section, section_to_rating_assignments, assignment_selection, data_btn, ratings_counts_all, ratings_counts):
    for i, pre_post in enumerate(['Pre', 'Post']):
        if('ALL' in assignment_selection):
            search = assignment_selection.replace(' ALL', '')

            for x in section_to_rating_assignments[section].keys():
                if search in x:
                    assignment_selection = x
                    break
    
        assignment = section_to_rating_assignments[section][assignment_selection][i]

        if(pre_post not in ratings_counts_all):
            ratings_counts_all[pre_post] = {
                'Count': 0,
                'Technology':{
                    'Total': 0,
                    'Low': {
                        'Number': 0,
                        'Students': {}
                    },
                    'Medium': {
                        'Number': 0,
                        'Students': {}
                    },
                    'High': {
                        'Number': 0,
                        'Students': {}
                    }
                },
                'Market':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
                'Organization':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
            }

        ratings = (
            InnovationRatings
                .select()
                .where(InnovationRatings.section==section.canvas_id)
                .where(InnovationRatings.assignment==assignment.assignment_id)
        )
    
        for rating in ratings:
            ratings_counts_all[pre_post][rating.rating_type]['Total'] += 1
            ratings_counts_all[pre_post][rating.rating_type][rating.rating]['Number'] += 1
            ratings_counts_all[pre_post][rating.rating_type][rating.rating]['Students'][rating.student_id] = rating.justification
  
    students = dict()
    
    for i, pre_post in enumerate(['Pre', 'Post']):
        if(pre_post) not in ratings_counts:
            ratings_counts[pre_post] = {
                'Count': 0,
                'Technology':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
                'Market':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
                'Organization':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
            }

        assignment = section_to_rating_assignments[section][assignment_selection][i]
        ratings = (
            InnovationRatings
                .select()
                .where(InnovationRatings.section==section.canvas_id)
                .where(InnovationRatings.assignment==assignment.assignment_id)
        )
    
        for rating in ratings:
            if rating.student_id not in students:
                students[rating.student_id] = dict()
            if rating.rating_type not in students[rating.student_id]:
                students[rating.student_id][rating.rating_type] = dict()
            students[rating.student_id][rating.rating_type][pre_post] = rating
    

    for student_id, type_ratings in students.items():
        if len(type_ratings) == 3:
            for rating_type, ratings in type_ratings.items():
                if len(ratings) == 2:
                    for pre_post, rating in ratings.items():
                        ratings_counts[pre_post][rating.rating_type]['Total'] += 1
                        ratings_counts[pre_post][rating.rating_type][rating.rating]['Number'] += 1
                        ratings_counts[pre_post][rating.rating_type][rating.rating]['Students'][student_id] = rating.justification

    return ratings_counts_all, ratings_counts 

def get_justifications(rating_counts, pre_or_post, rating_type, rating_tier, selected_assignment_name, selected_section):
    label = '{}-discussion: {} - {}'.format(pre_or_post, rating_type, rating_tier)


    data = pd.DataFrame(columns=['name', label])

    student_ids = list(rating_counts[pre_or_post][rating_type][rating_tier]['Students'].keys())

    random.shuffle(student_ids)

    if pre_or_post is 'Pre':
        pre_post_index = 0
    else:
        pre_post_index = 1

    for student_id in student_ids:
        name = Students.get_by_id(student_id).name

        new_row = dict()
        new_row['name'] = name
        new_row[label] = rating_counts[pre_or_post][rating_type][rating_tier]['Students'][student_id]
        data = data.append(new_row, ignore_index=True)


    return data.to_dict('records')

def populateRatings(assignments, course_num, section, data_btn):
    ratings_counts_all = dict()
    ratings_counts = dict()
    students = dict()

    for i, assignment in enumerate(assignments):
        if i==0: pp_assignent = 'Pre'
        else: pp_assignent = 'Post'
        
        ratings_counts_all[pp_assignent] = {
            'Count':0,
            'Technology':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
            'Market':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
            'Organization':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
        }

        ratings_counts[pp_assignent] = {
            'Count':0,
            'Technology':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
            'Market':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
            'Organization':{'Total': 0, 'Low': {'Number': 0, 'Students': {}}, 'Medium': {'Number': 0, 'Students': {}}, 'High': {'Number': 0, 'Students': {}}},
        }
        
        post = requests.post(canvasCredentials.API_URL + "/api/v1/courses/" + str(course_num) + "/quizzes/" + str(assignment.quiz_id) + "/reports" + "?access_token=" + canvasCredentials.API_KEY,
                                    data='quiz_report[report_type]=student_analysis')
            
        post_dict = json.loads(post.content)
        
        # Check that the CSV is ready with the provided progress URL
        while True:
            progress = requests.get(post_dict['progress_url'] + "?access_token=" + canvasCredentials.API_KEY)
            progress_dict = json.loads(progress.content)
            
            if progress_dict['completion'] == 100.0:
                break
                
            # Wait 1s if not ready yet
            time.sleep(1)
            
        link = requests.get(canvasCredentials.API_URL + "/api/v1/courses/" + str(course_num) + "/quizzes/" + str(assignment.quiz_id) + "/reports/" + str() + "?access_token=" + canvasCredentials.API_KEY)    
        download_link = json.loads(link.content)[0]['file']['url']
        
        download = requests.get(download_link)
        data = pd.read_csv(StringIO(download.content.decode("utf-8") ))
        
        # data.to_csv('assignments/' + pp_assignent + '.csv', encoding='utf-8')

        for j, row in data.iterrows():
            if row['section'] != section.name:
                continue

            counter = 0

            isJustification = False
            for question, answer in row[6:-3].items():
                if answer is np.NaN:
                    answer = ''
                
                if(counter%2==1):
                    # Handle Ratings
                    if isJustification:
                        isJustification = False

                        assigned_id = ''
                        if pre_post == 'Pre':
                            assigned_id += '1'
                        else:
                            assigned_id += '2'

                        if rating_type == 'Technology':
                            assigned_id += '1'
                        elif rating_type == 'Market':
                            assigned_id += '2'
                        else:
                            assigned_id += '3'

                        assigned_id += str(row.id)
                        assigned_id += str(assignment.id)
                        assigned_id = int(assigned_id)

                        section = Sections.get(Sections.name==row[3])

                        if answer == '1-Low' or answer == '2-Medium' or answer == '3-High':
                            answer = ''

                        
                        ratings_counts_all[pre_post][rating_type]['Total'] += 1
                        ratings_counts_all[pre_post][rating_type][rating]['Number'] += 1
                        ratings_counts_all[pre_post][rating_type][rating]['Students'][row.id] = answer
                        ratings_counts_all[pp_assignent]['Count'] += 1
                       
                        if row.id not in students:
                            students[row.id] = dict()
                        if rating_type not in students[row.id]:
                            students[row.id][rating_type] = dict()
                        students[row.id][rating_type][pre_post] = {'rating': rating , 'rating_type': rating_type, 'justification': answer}

                    elif 'Your rating' in question:
                        if 'Organization' in question:
                            rating_type = 'Organization'
                        elif 'Technology' in question:
                            rating_type = 'Technology'
                        elif 'Market' in question:
                            rating_type = 'Market'

                        if '1-Low' in answer:
                            rating = 'Low'
                        elif '2-Medium' in answer:
                            rating = 'Medium'
                        elif '3-High' in answer:
                            rating = 'High'
                        else:
                            counter += 1
                            continue
                            
                        if "pre-discussion" in assignment.name:
                            pre_post = "Pre"
                        elif "post-discussion" in assignment.name:
                            pre_post= "Post"
                        
                        isJustification = True
                
                counter += 1


   
    for student_id, type_ratings in students.items():
        if len(type_ratings) == 3:
            for rating_type, ratings in type_ratings.items():
                if len(ratings) == 2:
                    for pre_post, rating in InnovationRatings.items():
                        ratings_counts[pre_post][rating['rating_type']]['Total'] += 1
                        ratings_counts[pre_post][rating['rating_type']][rating['rating']]['Number'] += 1
                        ratings_counts[pre_post][rating['rating_type']][rating['rating']]['Students'][student_id] = rating['justification']

    return ratings_counts_all, ratings_counts
        
def get_figure(rating_counts, pre_post, rating_type, units):
    y = []
    for x in ['Low', 'Medium', 'High']:
        if units=='Counts':
            y.append(rating_counts[pre_post][rating_type][x]['Number'])
        else:
            y.append(rating_counts[pre_post][rating_type][x]['Percent'])

    return {
        'data': [
            {'x': ['Low', 'Medium', 'High'], 'y': y, 'type': 'bar', 'marker': dict(color=colors)},
        ],
        'layout': {
            'title': rating_type if pre_post=='Pre' else '',
            'margin': {'t': 30, 'b': 30},
            'yaxis': {'range': [0, rating_counts['max_count']]} if units=='Counts' else {'tickformat': ',.0%', 'range': [0, rating_counts['max_percent']]},
        }
    }

def get_delta_figure(rating_counts, rating_type):
    deltas = rating_counts[rating_type + '_delta']

    delta_colors = []
    for x in deltas:
        if x > 0:
            delta_colors.append('#274C77')
        else:
            delta_colors.append('#A63446')

    return {
        'data': [
            {'x': ['Low', 'Medium', 'High'], 'y': deltas, 'type': 'bar', 'marker': dict(color=delta_colors)},
        ],
        
        'layout': {
            'margin': {'t': 0, 'b': 25},
            'yaxis': {'range': [-rating_counts['max_delta'], rating_counts['max_delta']]} ,
        
        },
    }

# Updates section state
@app.callback(Output('selectedSection', 'children'), [Input('section-dropdown', 'value')])
def update_selected_section(value):
    return value

# Updates case dropdown
@app.callback(Output('case-dropdown', 'options'), [Input('section-dropdown', 'value')])
def update_case_dropdown(value):
    if value is None:
        value = 0
    
    if(value==len(cases)):
        return [{'label': x + ' innovation ratings', 'value': x + ' ALL'} for x in ['IKEA', 'Iridium', 'athenahealth', 'BMW', 'Google', 'P&G', 'IBM', 'Unilever', 'Venmo', '3M', 'Airbnb']]

    cases_sorted = cases[value]
    cases_tuple = []

    for case in cases_sorted:
        if 'IKEA' in case:
            cases_tuple.append([0, case])
        elif 'Iridium' in case:
            cases_tuple.append([1, case])
        elif 'athenahealth' in case:
            cases_tuple.append([2, case])
        elif 'BMW' in case:
            cases_tuple.append([3, case])
        elif 'Google' in case:
            cases_tuple.append([4, case])
        elif 'P&G' in case:
            cases_tuple.append([5, case])
        elif 'IBM' in case:
            cases_tuple.append([6, case])
        elif 'Unilever' in case:
            cases_tuple.append([7, case])
        elif 'Venmo' in case:
            cases_tuple.append([8, case])
        elif '3M' in case:
            cases_tuple.append([9, case])
        elif 'Airbnb' in case:
            cases_tuple.append([10, case])
    
    cases_tuple.sort(key=lambda tup: tup[0])
    final = [z[1] for z in cases_tuple]

    return [{'label': y, 'value': y} for x, y in enumerate(final)]

# Gets ratings and stores in state
@app.callback(Output('ratings', 'children'), [Input('case-dropdown', 'value'), Input('case-dropdown', 'options')], [State('selectedSection', 'children'), State('refresh_count', 'children'),  State('data_button', 'children')])
def update_ratings(selectedCase, options, selected_section, old_refresh_count, data_btn):
    global db

    if selectedCase not in [x['label'] for x in options] and isinstance(selectedCase, int):
        return 'null'

    db.close()
    db.connect()

    if(isinstance(selectedCase, str) and 'ALL' in selectedCase):
        selected_sections = sections
    else:
        selected_sections = [sections[selected_section]]
    
    ratings_all = {}
    ratings_prepost = {}

    for section in selected_sections:
        ratings_all, ratings_prepost  = get_rating_counts(section, section_to_rating_assignments, selectedCase, data_btn, ratings_all, ratings_prepost)

    result = dict()
    for index, ratings in enumerate([ratings_all, ratings_prepost]):
        max_count = 0
        max_percent = 0
        for i in ['Pre', 'Post']:
            for j in ['Technology', 'Market', 'Organization']:
                for k in ['Low', 'Medium', 'High']:
                    if ratings[i][j][k]['Number']>max_count:
                        max_count = ratings[i][j][k]['Number']

                    if ratings[i][j]['Total'] != 0:
                        ratings[i][j][k]['Percent'] = ratings[i][j][k]['Number']/ratings[i][j]['Total']

                        if ratings[i][j][k]['Percent'] > max_percent:
                            max_percent = ratings[i][j][k]['Percent']
                    else:
                        ratings[i][j][k]['Percent'] = 0

        max_delta = 0

        for i in ['Technology', 'Market', 'Organization']:
            deltas = []
            deltas_percents = []

            for x in ['Low', 'Medium', 'High']:
                if ratings['Post'][i]['Total'] == 0:
                    deltas.append(0)
                else:
                    delta = ratings['Post'][i][x]['Number'] - ratings['Pre'][i][x]['Number']
                    deltas.append( delta )
                    if abs(delta) > max_delta:
                        max_delta = abs(delta)


            ratings[i + '_delta'] = deltas

        ratings['max_delta'] = max_delta
        ratings['max_count'] = max_count
        ratings['max_percent'] = max_percent

        if(index == 0):
            result['All'] = ratings
        else:
            result['Pre-Post'] = ratings

    return json.dumps(result)

@app.callback(Output('ratings_students', 'children'), [Input('ratings', 'children'), Input('data_button', 'children')] )
def update_ratings_students(ratings_raw, data):
    ratings = json.loads(ratings_raw)

    if ratings is None:
        return "{}"

    ratings = ratings[data]

    # {
    #   id: {'Technology': [{}, {}]}
    # }
    students_to_ratings = dict()

    students = []
    for i in ['Pre', 'Post']:
        for j in ['Technology', 'Market', 'Organization']:
            for k in ['Low', 'Medium', 'High']:
                for x, y in ratings[i][j][k]['Students'].items():
                    if x not in students_to_ratings:
                        students_to_ratings[x] = {}
                    
                    if j not in students_to_ratings[x]:
                        students_to_ratings[x][j] = {}


                    students_to_ratings[x][j][i] = {'Rating': k, 'Justification': y}

    return json.dumps(students_to_ratings)

# @app.callback(Output('refresh_count', 'children'), [Input('ratings', 'children')], [State('refresh_button', 'n_clicks')])
# def update_refresh_count(children, new_refresh_count):
#     if new_refresh_count is None:
#         return '0'
#     return str(new_refresh_count)

# Create callbacks for 6 bar graphs
def create_bar_update_callback(pre_or_post, rating_type):
    def callback(value, units, data):
        if value is None:
            return {}

        if value == 'null':
            return {
                'data': [
                    {'x': ['Low', 'Medium', 'High'], 'y': [0, 0, 0], 'type': 'bar', 'marker': dict(color=['#7971ea', '#3e4e88', '#1a2c5b'])},
                ],
                
                'layout': {
                    'title': rating_type,
                    'margin': {'t': 50, 'b': 50},
                    'yaxis': {'range': [0, 50]},
                },
            }
        rating_counts = json.loads(value)[data]

        return get_figure(rating_counts, pre_or_post, rating_type, units)
    return callback

output_elements = ['Pre-Technology', 'Pre-Market', 'Pre-Organization', 'Post-Technology', 'Post-Market','Post-Organization']
for output_element in output_elements:
    arg_split = output_element.split('-')
    dynamically_generated_function = create_bar_update_callback(arg_split[0], arg_split[1])
    app.callback(
        Output(output_element, 'figure'), 
        [Input('ratings', 'children'), Input('units_button', 'children'), Input('data_button', 'children')] 
    )(dynamically_generated_function)

# Create callbacks for 3 bar graphs
def create_delta_update_callback(rating_type):
    def callback(value, units, data):
        if value is None:
            return {}

        if value == 'null':
            return {
                'data': [
                    {'x': ['Low', 'Medium', 'High'], 'y': [0, 0, 0], 'type': 'bar', 'marker': dict(color=colors)},
                ],
                
                'layout': {
                    'margin': {'t': 0, 'b': 25},
                    'yaxis': {'range': [-5, 5]},
                
                },
            }

        rating_counts = json.loads(value)[data]


        return get_delta_figure(rating_counts, rating_type)
    return callback

output_elements = ['delta-Technology', 'delta-Market', 'delta-Organization']
for output_element in output_elements:
    arg_split = output_element.split('-')
    dynamically_generated_function = create_delta_update_callback(arg_split[1])
    app.callback(
        Output(output_element, 'figure'), 
        [Input('ratings', 'children'), Input('units_button', 'children'), Input('data_button', 'children')] 
    )(dynamically_generated_function)

# Handle bar graph clicking
@app.callback(Output('previous_clickData', 'children'),
              [Input('button', 'n_clicks'), last_clicked('Pre-Technology', 'Pre-Market', 'Pre-Organization', 'Post-Technology', 'Post-Market', 'Post-Organization')])
def update_onclick_callback(nclicks, last_clickdata):
    if last_clickdata is None: 
        return '[]'

    click_data = last_clickdata["last_clicked_data"]
    clicked_id = last_clickdata["last_clicked"]

    if click_data is None:
        return '[]'

    data = [clicked_id, click_data['points'][0]['x']]

    return json.dumps(data)

# Show table
@app.callback(Output('tableView', 'style'), [Input('previous_clickData', 'children')], [State('tableView', 'style')])
def update_tableview(data, style):
    if data == '[]':
        return {'display':'none'}

    if style=={'display':'none'}:
        return {}

    return {'display':'none'}

# Handle bar graph clicking
@app.callback(Output('previous_delta_clickData', 'children'),
              [Input('close_delta_button', 'n_clicks'), last_clicked('delta-Technology', 'delta-Market', 'delta-Organization')])
def update_onclickdelta_callback(nclicks, last_clickdata):
    if last_clickdata is None: 
        return '[]'

    click_data = last_clickdata["last_clicked_data"]
    clicked_id = last_clickdata["last_clicked"]

    if click_data is None:
        return '[]'

    data = [clicked_id, click_data['points'][0]['x']]

    return json.dumps(data)

# Show side-by-side table
@app.callback(Output('deltaTableView', 'style'), [Input('previous_delta_clickData', 'children')], [State('deltaTableView', 'style')])
def update_deltatableview(data, style):
    if data == '[]':
        return {'display':'none'}

    if style=={'display':'none'}:
        return {}

    return {'display':'none'}

@app.callback(Output('delta_data1', 'children'), 
    [Input('deltaTableView', 'style')], 
    [State('ratings_students', 'children'), State('previous_delta_clickData', 'children')]
)
def dd1(style, ratings_raw, data_raw):
    if data_raw == '[]':
        return '{}'
    
    data = json.loads(data_raw)
    rating_type = data[0].replace('delta-', '')
    rating = data[1]

    ratings_students = json.loads(ratings_raw)

    graph = pd.DataFrame(columns=['Name', 'Pre-Rating', 'Pre-Evidence', 'Post-Rating', 'Post-Evidence'])

    student_keys = list(ratings_students.keys())
    random.shuffle(student_keys)

    for student in student_keys:
        try:
            pre_rating = ratings_students[student][rating_type]['Pre']
            post_rating = ratings_students[student][rating_type]['Post']
        except KeyError:
            continue

        if pre_rating['Rating']!=rating and post_rating['Rating']==rating:
            graph = graph.append({
                'Name': Students.get_by_id(student).name, 
                'Pre-Rating': pre_rating['Rating'], 
                'Pre-Evidence':pre_rating['Justification'],
                'Post-Rating': post_rating['Rating'], 
                'Post-Evidence':post_rating['Justification'],
            }, ignore_index=True)

    return json.dumps(graph.to_dict('records'))

@app.callback(Output('delta_datatable1', 'data'), [Input('delta_data1', 'children')])
def update_side1(data_raw):
    data = json.loads(data_raw)
    return pd.DataFrame(data).to_dict('records')

@app.callback(Output('delta_datatable1', 'columns'), [Input('delta_data1', 'children')])
def update_side1columns(data):
    just = json.loads(data)
    columns = list(reversed([{"name": i, "id": i} for i in pd.DataFrame(just).columns]))
    for i, x in enumerate(columns):
        if(x["name"]=="Name"):
            columns.insert(0, columns.pop(columns.index(x)))
            break
    return columns

@app.callback(Output('delta_data2', 'children'), 
    [Input('deltaTableView', 'style')], 
    [State('ratings_students', 'children'), State('previous_delta_clickData', 'children')]
)
def dd2(style, ratings_raw, data_raw):
    if data_raw == '[]':
        return '{}'
    
    data = json.loads(data_raw)
    rating_type = data[0].replace('delta-', '')
    rating = data[1]

    ratings_students = json.loads(ratings_raw)

    graph = pd.DataFrame(columns=['Name', 'Pre-Rating', 'Pre-Evidence', 'Post-Rating', 'Post-Evidence'])

    student_keys = list(ratings_students.keys())
    random.shuffle(student_keys)

    for student in student_keys:
        try:
            pre_rating = ratings_students[student][rating_type]['Pre']
            post_rating = ratings_students[student][rating_type]['Post']
        except KeyError:
            continue

        if pre_rating['Rating']==rating and post_rating['Rating']!=rating:
            graph = graph.append({
                'Name': Students.get_by_id(student).name, 
                'Pre-Rating': pre_rating['Rating'], 
                'Pre-Evidence':pre_rating['Justification'],
                'Post-Rating': post_rating['Rating'], 
                'Post-Evidence':post_rating['Justification'],
            }, ignore_index=True)
            
    return json.dumps(graph.to_dict('records'))

@app.callback(Output('delta_datatable2', 'data'), [Input('delta_data2', 'children')])
def update_side2(data_raw):
    data = json.loads(data_raw)
    return pd.DataFrame(data).to_dict('records')

@app.callback(Output('delta_datatable2', 'columns'), [Input('delta_data2', 'children')])
def update_side2columns(data):
    just = json.loads(data)
    columns = list(reversed([{"name": i, "id": i} for i in pd.DataFrame(just).columns]))
    for i, x in enumerate(columns):
        if(x["name"]=="Name"):
            columns.insert(0, columns.pop(columns.index(x)))
            break
    return columns
   

@app.callback(Output('tab_label1', 'label'), 
    [Input('deltaTableView', 'style')], 
    [State('ratings_students', 'children'), State('previous_delta_clickData', 'children')]
)
def update_side1_title(style, ratings_raw, data_raw):
    data = json.loads(data_raw)

    if len(data)==0:
        return ''

    return 'Switched to %s: %s' % (data[0].replace('delta-', ''), data[1])

@app.callback(Output('tab_label2', 'label'), 
    [Input('deltaTableView', 'style')], 
    [State('ratings_students', 'children'), State('previous_delta_clickData', 'children')]
)
def update_side2_title(style, ratings_raw, data_raw):
    data = json.loads(data_raw)

    if len(data)==0:
        return ''

    return 'Switched from %s: %s' % (data[0].replace('delta-', ''), data[1])


@app.callback(Output('justification_data', 'children'), 
    [Input('tableView', 'style')], 
    [State('ratings', 'children'), State('previous_clickData', 'children'), State('case-dropdown', 'value'), State('selectedSection', 'children'), State('data_button', 'children')]
)
def update_jd(style, ratings_raw, data_raw, selected_assignment_name, selected_section_index, data):
    if style == {} and data_raw!='[]':
        ratings = json.loads(ratings_raw)[data]

        data = json.loads(data_raw)

        data_split = data[0].split('-')

        pre_or_post = data_split[0]
        rating_type = data_split[1]
        rating_tier = data[1]

        if(selected_section_index == len(sections)):
            selected_section = sections[0]
        else:
            selected_section = sections[selected_section_index]

        just = get_justifications(ratings, pre_or_post, rating_type, rating_tier, selected_assignment_name, selected_section)
        return json.dumps(just)

    return '{}'


@app.callback(Output('datatable', 'data'), [Input('justification_data', 'children')])
def update_table(data):
    data_dict = json.loads(data)
    return pd.DataFrame(data_dict).to_dict("rows")

@app.callback(Output('datatable', 'columns'), [Input('justification_data', 'children')])
def update_table_columns(data):
    just = json.loads(data)
    return list(reversed([{"name": i, "id": i} for i in pd.DataFrame(just).columns]))

@app.callback(Output('units_button', 'children'), 
    [Input('units_button', 'n_clicks')],
    [State('units_button', 'children')] 
)
def change_units(n_clicks, children):
    if n_clicks is None:
        return 'Counts'

    if children == 'Counts':
        return 'Percentages'
    else:
        return 'Counts'

@app.callback(Output('data_button', 'children'), 
    [Input('data_button', 'n_clicks')],
    [State('data_button', 'children')] 
)
def change_data(n_clicks, children):
    if n_clicks is None:
        return 'All'

    if children == 'All':
        return 'Pre-Post'
    else:
        return 'All'

