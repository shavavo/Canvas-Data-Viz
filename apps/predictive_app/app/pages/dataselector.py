from app import dash_app as app
from app import app as flask_app
from app import tab_selected_style, tab_style, tabs_styles
import dash
import visdcc

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

import app_data_manager as app_data

import pandas as pd
from pandas import ExcelWriter

import io
import base64

def explore(graph, result=[], current=''):
    if( current!='' ):
        result.append(current)
    
    if( len(graph.keys()) == 0 ):
        return

    for x in graph.keys():
        if(x=='children'):
            continue

        if current=='':
            spacer = ''
        else:
            spacer = '__'
            
        explore(graph[x], result, current + spacer + x)
    
    return result

def shorthand(sequences):
    if not sequences:
        return ''

    lower = sequences[0]

    split = []
    temp = [sequences[0]]

    for x in range(len(sequences)-1):
        x += 1

        if sequences[x-1] + 1 != sequences[x]:
            split.append(temp)
            temp = [sequences[x]]
        else:
            temp.append(sequences[x])

    split.append(temp)

    result = ''
    for x in split:
        if len(x)==1:
            result += ',' + str(x[0])
        else:
            result += ',{}-{}'.format(x[0],x[-1])

    return result[1:]

def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n, index=False, encoding='utf8')
        writer.save()

assignment_options = ['All Assignments'] + app_data.assignment_names[0:-3]

graph = {
    'Innovation Ratings': {
        'Back': {},
    }, 
    'Decision Points': {
        'Back': {},
    },
    'Personal Preference Survey': {
        'Back': {},
        'Questions': {},
        'Concentration': {},
        'Company': {}
    },
}

question_to_answer, pp_questions = app_data.get_pp()
button_ids = explore(graph)

class DataSelector():
    def __init__(self, DS_ID):
        self.DS_ID = DS_ID + '____'

        self.all_buttons = [html.Button(x.split('__')[-1], id=self.DS_ID + x, n_clicks_timestamp=0, className='data_category_btns', style={'display': 'none'}) for x in button_ids]
        self.all_inputs = [Input(self.DS_ID + x, 'n_clicks_timestamp') for x in button_ids]
        self.all_inputs.extend([
            Input(self.DS_ID + 'ratings_add', 'n_clicks_timestamp'),
            Input(self.DS_ID + 'decisionpoints_add', 'n_clicks_timestamp'),
            Input(self.DS_ID + 'pp_add', 'n_clicks_timestamp')
        ])

        self.assign_callbacks()
    

    def serve_layout(self, sess_id):
        download_link =  html.A(id=self.DS_ID + 'download-link', target='_blank', href='dimred/download?value=' + sess_id),

        layout = html.Div(children=[
            dcc.Store(id=self.DS_ID + 'session_id', data=sess_id, storage_type='memory'),
            dcc.Store(id=self.DS_ID + 'location', storage_type='memory'),
            dcc.Store(id=self.DS_ID + 'currently_selected_data', storage_type='memory'),
            dcc.Store(id=self.DS_ID + 'all_selected_pool', storage_type='memory'),
            dcc.Store(id=self.DS_ID + 'all_selected_data', storage_type='memory'),
            dcc.Store(id=self.DS_ID + 'uploaded_data', storage_type='memory'),

            visdcc.Run_js(id=self.DS_ID + 'javascript'),

            html.Div(children=[
                html.Div(children=[
                    html.Button('Done', id=self.DS_ID + 'close_selector', className="close_selector", n_clicks_timestamp=0),
                    html.H4('Select Data:', id=self.DS_ID + 'path', style={'textAlign': 'center', 'margin': '0', 'paddingTop': '3rem'}),
                    html.Div(children=[
                        html.Div(
                            id=self.DS_ID + 'button_container', 
                            children=self.all_buttons, 
                        ),
                        html.Div(children=[
                            html.Div(children=[
                                dcc.Dropdown(
                                    id=self.DS_ID + 'ratings_assignment',
                                    options=[{'label': x, 'value': x} for x in assignment_options],
                                    style={'width': '25vw'},
                                    multi=True,
                                    placeholder='Select Assignment(s)'
                                ),
                                dcc.Dropdown(
                                    id=self.DS_ID + 'ratings_prepost',
                                    options=[{'label': x, 'value': x} for x in ['All Pre/Post', 'Pre', 'Post']],
                                    style={'width': '25vw'},
                                    multi=True,
                                    placeholder='Select Pre/Post'
                                ),
                                dcc.Dropdown(
                                    id=self.DS_ID + 'ratings_type',
                                    options=[{'label': x, 'value': x} for x in ['All Types', 'Technology', 'Market', 'Organization']],
                                    style={'width': '25vw'},
                                    multi=True,
                                    placeholder='Select Type'
                                ),
                                html.Button('Add', id=self.DS_ID + 'ratings_add', n_clicks_timestamp=0)
                            ], id=self.DS_ID + 'ratings_selectors', style={'display': 'flex'}),
                            html.Div(children=[
                                dcc.Dropdown(
                                    id=self.DS_ID + 'decisionpoints_assignment',
                                    options=[{'label': x, 'value': x} for x in [assignment_options[0]] + assignment_options[2:]],
                                    style={'width': '25vw'},
                                    multi=True,
                                    placeholder='Select Assignment(s)'
                                ),
                                dcc.Dropdown(
                                    id=self.DS_ID + 'decisionpoints_prepost',
                                    options=[{'label': x, 'value': x} for x in ['All Pre/Post', 'Pre', 'Post']],
                                    style={'width': '25vw'},
                                    multi=True,
                                    placeholder='Select Pre/Post'
                                ),
                                html.Button('Add', id=self.DS_ID + 'decisionpoints_add', n_clicks_timestamp=0)
                            ], id=self.DS_ID + 'decisionpoints_selectors', style={'display': 'flex'}),
                            html.Div(children=[
                                html.Button('Add', id=self.DS_ID + 'pp_add', n_clicks_timestamp=0)
                            ], id=self.DS_ID + 'pp_selectors', style={'display': 'flex'}),

                            


                        ], id=self.DS_ID + 'selector_container')
                    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'center'}),
                ]),
                
                html.Div(children=[
                    html.Div(children=[
                        html.Div(children=[
                            html.H4('Currently Selected Data', style={'textAlign':'center'}),
                            html.Div(children=[
                                html.P('n=_, _ dropped', id=self.DS_ID + 'current_description')
                            ], style={'height': '5vh', 'textAlign': 'center'}),
                            html.Div(children=[
                                    dash_table.DataTable(
                                        id=self.DS_ID + 'currently_selected',
                                        style_table={'height': '100%', 'overflowY': 'scroll'},
                                        page_current = 0,
                                        page_size = 8,
                                        page_action='custom',
                                        fill_width = True
                                    )
                            ], style={'flex': '1', 'minHeight': '0'})

                        ], style={'display': 'flex', 'flexDirection': 'column', 'width': '48%', 'margin': '1%'}),

                        html.Div(children=[
                            download_link[0],
                            dcc.Upload(id=self.DS_ID + 'upload_all_selected', children=[html.Button('Upload', className='downloadButton', style={'left': '51%'})]),
                            html.Button(id=self.DS_ID + 'download_all_selected', children=['Download'], className='downloadButton'),
                            html.H4('All Selected Data', style={'textAlign':'center'}),
                            html.P('n = 0', id=self.DS_ID + 'all_selected_n', style={'textAlign': 'center'}),

                            dcc.Dropdown(id=self.DS_ID + 'label', className='dropdown-spacing', placeholder='Label', style={} if (self.DS_ID=='class____' or self.DS_ID=='reg____') else {'display': 'none'}),

                            dcc.Dropdown(
                                id=self.DS_ID + 'all_selected_dropdown',
                                className='dropdown-spacing',
                                multi=True,
                                value=[],
                                placeholder='Data Sets'
                            ),

                     
                            html.Div(children=[
                                dash_table.DataTable(
                                    id=self.DS_ID + 'all_selected',
                                    style_table={'height': '100%', 'overflowY': 'scroll'},
                                    page_current = 0,
                                    page_size = 6,
                                    page_action='custom',
                                    merge_duplicate_headers=True,
                                    fill_width = True

                        
                                )
                            ], style={'flex': '1', 'minHeight': '0'})
                         
                                
                        ], style={'display': 'flex', 'flexDirection': 'column', 'width': '48%', 'margin': '1%'}),   
                    ], style={'display': 'flex', 'height': '100%'}),

                    

                ], style={'flex': '1', 'marginTop': '1rem', 'minHeight': '0'})
            ], id=self.DS_ID + 'selector', className="selector", style={'display': 'none'}),

            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        html.H4("Select Questions", style={'marginRight': '2rem'}),
                        html.Button("Select All", id=self.DS_ID + 'select_all_pp', n_clicks_timestamp=0),
                        html.Button("Deselect All", id=self.DS_ID + 'deselect_all_pp', n_clicks_timestamp=0),
                        html.Button('Close', id=self.DS_ID + 'close_pp_selector', n_clicks_timestamp=0, style={'marginLeft': 'auto'}),
                    ], style={'display': 'flex', 'paddingBottom': '2rem'}),
                    
                    dash_table.DataTable(
                        id=self.DS_ID + 'pp_table',
                        columns=[{"name": 'Questions', "id": 'Questions'}],
                        data=[{'Questions': x} for x in pp_questions],
                        style_cell={
                            'textAlign': 'left',
                        },
                        style_table={
                            'maxHeight': '80vh',
                            'overflowY': 'scroll',
                        },
                        style_data={'whiteSpace': 'normal'},
                        css=[{
                            'selector': '.dash-cell div.dash-cell-value',
                            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                        }],
                        row_selectable="multi",
                        filter_action='native',
                        sort_action='native',
                        fill_width = True

                    )
                ], className='pp_container'),  
            ], id=self.DS_ID + 'pp_questions_selector', className='pp_questions', style={'display': 'none'}),

            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        html.H4("Select Concentrations", style={'marginRight': '2rem'}),
                        html.Button("Select All", id=self.DS_ID + 'select_all_concentration', n_clicks_timestamp=0),
                        html.Button("Deselect All", id=self.DS_ID + 'deselect_all_concentration', n_clicks_timestamp=0),
                        html.Button('Close', id=self.DS_ID + 'close_concentration_selector', n_clicks_timestamp=0, style={'marginLeft': 'auto'}),
                    ], style={'display': 'flex', 'paddingBottom': '2rem'}),
                    
                    dash_table.DataTable(
                        id=self.DS_ID + 'concentration_table',
                        columns=[{"name": app_data.additional_questions[1][0], "id": app_data.additional_questions[1][0]}],
                        data=[{app_data.additional_questions[1][0]: x} for x in app_data.concentrations],
                        style_cell={
                            'textAlign': 'left',
                        },
                        style_table={
                            'maxHeight': '80vh',
                            'overflowY': 'scroll',
                        },
                        style_data={'whiteSpace': 'normal'},
                        css=[{
                            'selector': '.dash-cell div.dash-cell-value',
                            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                        }],
                        row_selectable="multi",
                        filter_action='native',
                        sort_action='native',
                        fill_width = True

                    )
                ], className='pp_container'),  
            ], id=self.DS_ID + 'concentration_selector', className='pp_questions', style={'display': 'none'}),

            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        html.H4("Select Companies", style={'marginRight': '2rem'}),
                        html.Button("Select All", id=self.DS_ID + 'select_all_companies', n_clicks_timestamp=0),
                        html.Button("Deselect All", id=self.DS_ID + 'deselect_all_companies', n_clicks_timestamp=0),
                        html.Button('Close', id=self.DS_ID + 'close_companies_selector', n_clicks_timestamp=0, style={'marginLeft': 'auto'}),
                    ], style={'display': 'flex', 'paddingBottom': '2rem'}),
                    
                    dash_table.DataTable(
                        id=self.DS_ID + 'companies_table',
                        columns=[{"name": app_data.additional_questions[0][0], "id": app_data.additional_questions[0][0]}],
                        data=[{app_data.additional_questions[0][0]: x} for x in app_data.companies],
                        style_cell={
                            'textAlign': 'left',
                        },
                        style_table={
                            'maxHeight': '80vh',
                            'overflowY': 'scroll',
                        },
                        style_data={'whiteSpace': 'normal'},
                        css=[{
                            'selector': '.dash-cell div.dash-cell-value',
                            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                        }],
                        row_selectable="multi",
                        filter_action='native',
                        sort_action='native',
                        fill_width = True

                    )
                ], className='pp_container'),  
            ], id=self.DS_ID + 'companies_selector', className='pp_questions', style={'display': 'none'}),

        ], style={'position':'absolute', 'width':'100%', 'top':'5vh'})


        return layout

    def assign_callbacks(self):
        @app.callback(Output(self.DS_ID + 'selector', 'style'), [Input(self.DS_ID + 'open_selector', 'n_clicks_timestamp'), Input(self.DS_ID + 'close_selector', 'n_clicks_timestamp')])
        def show_selector(btn1, btn2):
            if( int(btn1) <= int(btn2) ):
                return {'display': 'none'}
            return {}


        @app.callback(Output(self.DS_ID + 'location', 'data'), self.all_inputs + [Input(self.DS_ID + 'close_pp_selector', 'n_clicks_timestamp'), Input(self.DS_ID + 'close_concentration_selector', 'n_clicks_timestamp'), Input(self.DS_ID + 'close_companies_selector', 'n_clicks_timestamp')])
        def update_data(*timestamps):
            if( all(x==0 for x in timestamps) ):
                location = prefix = ''
                current = graph
            else:
                max_value = max(timestamps)
                max_index = timestamps.index(max_value)

                if max_index == len(timestamps)-1 or max_index == len(timestamps)-2 or max_index == len(timestamps)-3:
                    location = 'Personal Preference Survey'
                elif(max_index >= len(button_ids)):
                    location = ''
                else:
                    location = button_ids[max_index]

                if 'Back' in location:
                    location = '__'.join( location.split('__')[0:-2] )
                
                if 'All Questions' in location:
                    location = '__'.join( location.split('__')[0:-1] )
                

                prefix = ''
                current = graph
                
                if(location!=''):
                    for x in location.split('__'):
                        prefix += x + '__'
                        current = current[x]

            ids_to_show = {}
            for x in current:
                ids_to_show[prefix + x] = None

            return {
                'location': location,
                'show': ids_to_show
            }

        def create_callback(btn_id):
            def callback(data):
                ids_to_show = data['show']
                if btn_id in ids_to_show:
                    return {}
                else:
                    return {'display': 'none'}

                
            return callback

        for btn_id in button_ids:
            app.callback(Output(self.DS_ID + btn_id, 'style'), [Input(self.DS_ID + 'location', 'data')])(create_callback(btn_id))

        @app.callback(Output(self.DS_ID + 'uploaded_data', 'data'), [Input(self.DS_ID + 'upload_all_selected', 'contents')])
        def update_uploaded_data(contents):
            if not contents:
                return {}
            
            _, content_string = contents.split(',')

            decoded = base64.b64decode(content_string)
            stream = io.BytesIO(decoded)
            sheet1 = pd.read_excel(stream, 'sheet0')
            sheet2 = pd.read_excel(stream, 'sheet1')

            return {'sheet1': sheet1.to_dict(), 'sheet2': sheet2.to_dict()}


        @app.callback(Output(self.DS_ID + 'javascript', 'run'), [Input(self.DS_ID + 'download_all_selected', 'n_clicks')], [State(self.DS_ID + 'all_selected_dropdown', 'value'), State(self.DS_ID + 'all_selected_pool', 'data'), State(self.DS_ID + 'session_id', 'data')])
        def save_all_selected(n_clicks, selected, data_pool, sess_id):
            if(selected == []):
                return {}

            data_sets = []
            sheet2 = pd.DataFrame(columns=['selected', 'columns'])

            for x in selected:
                data = data_pool[x]
                df = pd.DataFrame.from_dict(data)
                df = df.set_index('student')
                data_sets.append(df)

                sheet2 = sheet2.append({'selected': x, 'columns': '; '.join(df.columns)}, ignore_index=True)

            sheet1 = pd.concat(data_sets, axis=1, join='inner').reset_index()

            save_xls([sheet1, sheet2], 'temp/{}.xlsx'.format(sess_id))

            return 'document.getElementById("{}download-link").click();'.format(self.DS_ID)

        


        # @app.callback(Output(self.DS_ID + 'pp_table', 'selected_rows'), [Input(self.DS_ID + 'select_all_pp', 'n_clicks_timestamp'), Input(self.DS_ID + 'deselect_all_pp', 'n_clicks_timestamp'), Input(self.DS_ID + 'pp_add', 'n_clicks_timestamp')], [State(self.DS_ID + 'pp_table', 'data')])
        def select_all_pp(select, deselect, add, data):
            if int(select) <= int(deselect) or int(select) <= int(add):
                return []

            return list(range(len(data)))

        for x in ['pp', 'concentration', 'companies']:
            app.callback(
                Output(self.DS_ID + x + '_table', 'selected_rows'), 
                [Input(self.DS_ID + 'select_all_' + x, 'n_clicks_timestamp'), Input(self.DS_ID + 'deselect_all_' + x, 'n_clicks_timestamp'), Input(self.DS_ID + 'pp_add', 'n_clicks_timestamp')], 
                [State(self.DS_ID + x + '_table', 'data')]
            )(select_all_pp)


        def create_show_selector_callback(match):
            def show_selector(data):
                location = data['location']
                
                if match in location:
                    return {'zIndex': 2}

                return {'display': 'none'}

            return show_selector

        selectors = [
            {'id': 'pp_questions_selector', 'match': 'Questions'},
            {'id': 'concentration_selector', 'match': 'Concentration'},
            {'id': 'companies_selector', 'match': 'Company'}
        ]

        for x in selectors:
            app.callback(Output(self.DS_ID + x['id'], 'style'), [Input(self.DS_ID + 'location', 'data')])(create_show_selector_callback(x['match']))


        @app.callback(Output(self.DS_ID + 'ratings_selectors', 'style'), [Input(self.DS_ID + 'location', 'data')])
        def hide_ratings_selectors(data):
            location = data['location']
            if(location == 'Innovation Ratings'):
                return {'display': 'flex'}
            return {'display': 'none'}

        @app.callback(Output(self.DS_ID + 'decisionpoints_selectors', 'style'), [Input(self.DS_ID + 'location', 'data')])
        def hide_decisionpoints_selectors(data):
            location = data['location']
            if(location == 'Decision Points'):
                return {'display': 'flex'}
            return {'display': 'none'}

        @app.callback(Output(self.DS_ID + 'pp_selectors', 'style'), [Input(self.DS_ID + 'location', 'data')])
        def hide_pp_selectors(data):
            location = data['location']
            if(location == 'Personal Preference Survey'):
                return {'display': 'flex'}
            return {'display': 'none'}

        @app.callback(Output(self.DS_ID + 'current_description', 'children'), [Input(self.DS_ID + 'currently_selected_data', 'data')])
        def update_desc(data):
            if data=={}:
                return ''
            return 'n={}, {} dropped'.format(data['n'], data['max'] - data['n'])

        def create_callback2():
            def callback(n_clicks):
                return []
            return callback

        for x in ['ratings_assignment', 'ratings_prepost', 'ratings_type']:
            app.callback(Output(self.DS_ID + x, 'value'), [Input(self.DS_ID + 'ratings_add', 'n_clicks')])(create_callback2())

        for x in ['decisionpoints_assignment', 'decisionpoints_prepost']:
            app.callback(Output(self.DS_ID + x, 'value'), [Input(self.DS_ID + 'ratings_add', 'n_clicks')])(create_callback2())

        @app.callback(Output(self.DS_ID + 'path', 'children'), [Input(self.DS_ID + 'location','data')])
        def update_path(data):
            location = data['location']

            if(location==''):
                return 'Select Data:'

            return location.replace('__', '/')


        @app.callback(Output(self.DS_ID + 'currently_selected_data', 'data'), 
            [
                Input(self.DS_ID + 'ratings_assignment', 'value'), Input(self.DS_ID + 'ratings_prepost', 'value'), Input(self.DS_ID + 'ratings_type', 'value'),
                Input(self.DS_ID + 'decisionpoints_assignment', 'value'), Input(self.DS_ID + 'decisionpoints_prepost', 'value'),
                Input(self.DS_ID + 'pp_table', 'selected_rows'),
                Input(self.DS_ID + 'concentration_table', 'selected_rows'),
                Input(self.DS_ID + 'companies_table', 'selected_rows'),
                Input(self.DS_ID + 'location', 'data')
            ],
            [
                State(self.DS_ID + 'pp_table', 'data'), State(self.DS_ID + 'concentration_table', 'data'), State(self.DS_ID + 'companies_table', 'data'),
                State(self.DS_ID + 'close_pp_selector', 'n_clicks_timestamp'), State(self.DS_ID + 'close_concentration_selector', 'n_clicks_timestamp'), State(self.DS_ID + 'close_companies_selector', 'n_clicks_timestamp'),
            ]
        )
        def update_currently_selected(ratings_assignment, ratings_prepost, ratings_type, decisionpoints_assignment, decisionpoints_prepost, pp_selected, concentration_selected, companies_selected, data, pp_data, concentration_data, companies_data, pp_click, concentration_click, companies_click):
            app_data.db.close()
            app_data.db.connect()
            
            location = data['location']

            if(location=='Innovation Ratings' and ratings_assignment and ratings_prepost and ratings_type):
                if('All Assignments' in ratings_assignment):
                    X = assignment_options[1:]
                    name1 = 'all'
                else:
                    X = ratings_assignment
                    name1 = ','.join(ratings_assignment)
                
                if('All Pre/Post' in ratings_prepost):
                    Y = ['pre', 'post']
                    name2 = 'pre,post'
                else:
                    Y = [x.lower() for x in ratings_prepost]
                    name2 = ','.join(Y)
                
                if('All Types' in ratings_type):
                    Z = ['Technology', 'Market', 'Organization']
                    name3 = ','.join(Z)
                else:
                    Z = ratings_type
                    name3 = ','.join(Z)

                name = 'IR: ' + '_'.join([name1, name2, name3])
                data_sets = []
                data_set_names = []
            
                for x in X:
                    for y in Y:
                        for z in Z:
                            data = app_data.get_student_to_ratings(x, y, z)
                            data_sets.append(data)
                            data_set_names.append('{}_{}_{}'.format(x, y, z))
            elif(location=='Decision Points' and decisionpoints_assignment and decisionpoints_prepost):
                if('All Assignments' in decisionpoints_assignment):
                    X = assignment_options[2:]
                    name1 = 'all'
                else:
                    X = decisionpoints_assignment
                    name1 = ','.join(X)

                if('All Pre/Post' in decisionpoints_prepost):
                    Y = ['pre', 'post']
                    name2 = 'pre,post'
                else:
                    Y = [x.lower() for x in decisionpoints_prepost]
                    name2 = ','.join(Y)

                name = 'DP: ' + '_'.join([name1, name2])
                data_sets = []
                data_set_names = []

                for x in X:
                    for y in Y:
                        question, data, _, _ = app_data.get_student_to_dp(x, y)
                        data_sets.append(data)
                        data_set_names.append('DP_{}_{}: {}'.format(y, x, question))
            elif(location=='Personal Preference Survey'):
                timestamps = [pp_click, concentration_click, companies_click]
                max_value = max(timestamps)
                max_index = timestamps.index(max_value)

                data_sets = []
                data_set_names = []

                if max_index==0:            
                    if not pp_selected:
                        return {}

                    name = 'Personal Preference Survey Q' + shorthand(pp_selected)

                    for x in pp_selected:
                        selected_question = pp_data[x]['Questions']
                        data_sets.append(question_to_answer[selected_question])
                        data_set_names.append('Q{}: '.format(x) + selected_question)
                elif max_index==1:
                    if not concentration_selected:
                        return {}
                    
                    name = 'Concentrations ' + shorthand(concentration_selected)
                    question = app_data.additional_questions[1][0]

                    for x in concentration_selected:
                        concentration = concentration_data[x][question]
                        y = question_to_answer[question][concentration]
                        data_sets.append(y)
                        data_set_names.append('Concentration: ' + concentration)
                elif max_index==2:
                    if not companies_selected:
                        return {}
                    
                    name = 'Companies ' + shorthand(companies_selected)
                    question = app_data.additional_questions[0][0]

                    for x in companies_selected:
                        company = companies_data[x][question]
                        y = question_to_answer[question][company]
                        data_sets.append(y)
                        data_set_names.append('Company: ' + company)
                
            else:
                return {}

            intersect = set(data_sets[0].keys())
            maximum = len(intersect)
            for data in data_sets:
                keys = set(data.keys())
                maximum = max(len(keys), maximum)
                intersect = intersect & keys

            records = []
            for x in intersect:
                new_record = {'student': x}
                for index, data in enumerate(data_sets):
                    new_record[data_set_names[index]] = data[x]
                records.append(new_record)
            
            result = pd.DataFrame.from_records(records)
            result.set_index('student').reset_index()

            return {'data': result.to_dict(), 'n': result.shape[0], 'max': maximum, 'name': name }

        @app.callback(Output(self.DS_ID + 'currently_selected', 'columns'), [Input(self.DS_ID + 'currently_selected_data', 'data')])
        def update_currently_selected_columns(data):
            if('data' not in data):
                return []

            df = pd.DataFrame.from_dict(data['data'])

            columns = list(df.columns)
            columns.reverse()

            return [{"name": i, "id": i} for i in columns]

        @app.callback(Output(self.DS_ID + 'currently_selected', 'data'), [Input(self.DS_ID + 'currently_selected_data', 'data'), Input(self.DS_ID + 'currently_selected', 'page_current'), Input(self.DS_ID + 'currently_selected', 'page_size')])
        def update_currently_selected_data(data, page_current, page_size):
            if('data' not in data):
                return []

            df = pd.DataFrame.from_dict(data['data'])
            
            return df.iloc[
                page_current*page_size:
                (page_current + 1)*page_size
            ].to_dict('records')

        @app.callback(
            Output(self.DS_ID + 'all_selected_pool', 'data'), 
            [Input(self.DS_ID + 'ratings_add', 'n_clicks_timestamp'), Input(self.DS_ID + 'decisionpoints_add', 'n_clicks_timestamp'), Input(self.DS_ID + 'pp_add', 'n_clicks_timestamp'), Input(self.DS_ID + 'uploaded_data', 'data')],
            [State(self.DS_ID + 'currently_selected_data', 'data'), State(self.DS_ID + 'all_selected_pool', 'data')]
        )
        def update_all_selected_pool(ratings_add, dp_add, pp_add, uploaded_data, data, all_data):
            # if ratings_add==0 and dp_add==0 and pp_add==0 and uploaded_data==None:
            #     return {}
            if not uploaded_data and not data and not all_data:
                return {}

            if uploaded_data:
                sheet1 = pd.DataFrame.from_dict(uploaded_data['sheet1'])
                sheet2 = pd.DataFrame.from_dict(uploaded_data['sheet2'])

                for index, row in sheet2.iterrows():
                    key = row['selected']
                    columns = row['columns'].split('; ') + ['student']
                    all_data[key] = sheet1[columns].to_dict()

                all_data['new'] = '; '.join(sheet2['selected'])

            if('name' not in data):
                return all_data

            name = data['name']
            all_data['new'] = name
            all_data[name] = data['data']

            return all_data

        @app.callback(Output(self.DS_ID + 'all_selected_dropdown', 'options'), [Input(self.DS_ID + 'all_selected_pool', 'data')])
        def update_all_selected_options(data):
            keys = list(data.keys())
            if 'new' in keys:
                keys.remove('new')
            return [{"label": i, "value": i} for i in keys]
            
        @app.callback(Output(self.DS_ID + 'all_selected_dropdown', 'value'), 
            [Input(self.DS_ID + 'all_selected_pool', 'data')],
            [State(self.DS_ID + 'all_selected_dropdown', 'value')]
        )
        def update_all_selected_dropdown_value(data, old_value):
            if data is None:
                return old_value
            if 'new' not in data:
                return old_value
            old_value += data['new'].split('; ')
            return old_value

        @app.callback(Output(self.DS_ID + 'all_selected_data', 'data'), [Input(self.DS_ID + 'all_selected_dropdown', 'value')], [State(self.DS_ID + 'all_selected_pool', 'data')])
        def update_all_selected_data(selected, data_pool):
            if(selected == []):
                return {}

            data_sets = []
            columns = []

            for x in selected:
                data = data_pool[x]
                df = pd.DataFrame.from_dict(data)
                df = df.set_index('student')
                data_sets.append(df)
                for y in df.columns:
                    columns.append('{}____{}'.format(x, y))

            result = pd.concat(data_sets, axis=1, join='inner').reset_index()
            
            return {'data': result.to_dict(), 'n': result.shape[0], 'columns': columns}

        @app.callback(Output(self.DS_ID + 'all_selected', 'columns'), [Input(self.DS_ID + 'all_selected_data', 'data')])
        def update_all_selected_columns(data):
            if('data' not in data):
                return []

            df = pd.DataFrame.from_dict(data['data'])
            columns = list(df.columns)
            return [{"name": i, "id": i} for i in columns]
            # columns = data['columns']
            # return [{"name": x.split('____'), "id": x.split('____')[-1]} for x in columns]

        @app.callback(Output(self.DS_ID + 'all_selected', 'data'), [Input(self.DS_ID + 'all_selected_data', 'data'), Input(self.DS_ID + 'all_selected', 'page_current'), Input(self.DS_ID + 'all_selected', 'page_size')])
        def update_all_selected_table(data, page_current, page_size):
            if('data' not in data):
                return []

            df = pd.DataFrame.from_dict(data['data'])
            return df.iloc[
                page_current*page_size:
                (page_current + 1)*page_size
            ].to_dict('records')

        @app.callback(Output(self.DS_ID + 'all_selected_n', 'children'), [Input(self.DS_ID + 'all_selected_data', 'data')])
        def update_all_selected_n(data):
            if 'n' not in data:
                return ''
            return 'n = {}'.format(data['n'])