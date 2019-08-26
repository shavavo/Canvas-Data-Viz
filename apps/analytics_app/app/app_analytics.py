from app import dash_app as app

import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from plotly import tools
from scipy import stats

import canvasModels
from canvasModels import *
db = canvasModels.init_db(sqlCredentials)

from collections import Counter

colors = ['#e5e523', '#1f9274', '#36226a']
default_colors = [None, '#1f77b4', 
    '#ff7f0e',  
    '#2ca02c', 
    '#d62728',  
    '#9467bd',  
    '#8c564b',  
    '#e377c2',  
    '#7f7f7f',  
    '#bcbd22',  
    '#17becf'   
    ]

assignments = Assignments.select()
assignment_names = ['IKEA', 'Iridium', 'athenahealth', 'BMW', 'Google', 'P&G', 'IBM', 'Unilever', 'Venmo', '3M', 'Airbnb', 'Personal Survey Questions', 'Concentration', 'Type of Company']
assignment_to_ratingIDs = dict()

for assignment in assignments:
    if 'innovation ratings' in assignment.name:
        company = assignment.name.split("pre-discussion")[0].split("post-discussion")[0].strip().split()[-1]
        pre_post = 'Pre' if 'pre-discussion' in assignment.name else 'Post'

        if pre_post == 'Post':
            continue
        
        if company not in assignment_to_ratingIDs:
            assignment_to_ratingIDs[company] = dict()
        if pre_post not in assignment_to_ratingIDs[company]:
            assignment_to_ratingIDs[company][pre_post] = []
            
        assignment_to_ratingIDs[company][pre_post].append(assignment)
        
        
        

pp_assignments = list(Assignments.select().where(Assignments.name.contains("Personal Preference Survey")))
pp_questions = []
pp_table = pd.DataFrame(columns=['Questions'])


question_to_answer = dict()

for assignment in pp_assignments:
    responses = QuizResponses.select().where(QuizResponses.assignment==assignment)
    
    for response in responses:
        cleaned_question = response.question.split(": ")[1]
        
        if cleaned_question not in question_to_answer:
            question_to_answer[cleaned_question] = dict()
        
        
        question_to_answer[cleaned_question][response.student_id] = response.answer
        

for question, student_to_answer in question_to_answer.items():
    counts = dict(Counter(student_to_answer.values()))
    if len(counts) > 20 or len(counts)==1:
        continue
    pp_questions.append(question)
    pp_table = pp_table.append({'Questions': question}, ignore_index=True)
        

concentrations = { 
    'Decision Sciences', 
    'Energy and Environment', 
    'Entrepreneurship and Innovation', 
    'Finance', 
    'Financial Analysis',
    'Health Sector Management',
    'Leadership and Ethics',
    'Management',
    'Marketing',
    'Operations Management',
    'Social Entrepreneurship',
    'Strategy'
 }

companies = {
    'Advertising / Marketing',
    'Automotive / Transportation',
    'Clothing / Apparel',
    'Consumer Electronics',
    'Consumer Packaged Goods',
    'Education / Training',
    'Energy / Power',
    'Entertainment / Media',
    'Financial Services',
    'Food / Beverage',
    'Hardware / Computing',
    'Health Care / Medical Devices',
    'Manufacturing',
    'Online Business / E-Commerce',
    'Professional Services',
    'Retail / Shopping',
    'Social Sector / Non-Profit',
    'Software / Apps',
    'Sports / Athletics',
    'Travel / Hospitality'
}

def binarize_question(question_to_answer, all_answers, question):
    answer_to_students = {}
    
    for student, answer in question_to_answer[question].items():
        temp_answer = set()
        for x in answer.split(','):
            temp_answer.add(x)
            
            if(x not in answer_to_students):
                answer_to_students[x] = {}
            answer_to_students[x][student] = '1 - yes'
            
        for x in all_answers-temp_answer:
            if(x not in answer_to_students):
                answer_to_students[x] = {}
            answer_to_students[x][student] = '0 - no'
    
    return answer_to_students
        
company_question = "What type of company are you interested in studying for the final project in this course (check all that apply)?"
company_to_students = binarize_question(question_to_answer, companies, company_question)

concentration_question = "What is your current business school concentration (check all that apply)?"
concentration_to_students = binarize_question(question_to_answer, concentrations, concentration_question)

concentration_table = pd.DataFrame(list(concentration_to_students.keys()))
concentration_table.columns = ['Concentrations']


company_table = pd.DataFrame(list(company_to_students.keys()))
company_table.columns = ['Company Interests']








full_options = ['Technology Ratings', 'Market Ratings', 'Organization Ratings', 'Decision Point']

layout = html.Div(children=[
    html.Div([
        dcc.Dropdown(
            id='x',
            options=[
                {'label': x, 'value': x} for x in list(assignment_names)
            ],
        ),
        dcc.Dropdown(
            id='y',
            options=[
                {'label': x, 'value': x} for x in list(assignment_names)
            ],
        ),
        dcc.Dropdown(
            id='z',
            options=[
                {'label': x, 'value': x} for x in list(assignment_names)
            ],
        ),
        dcc.Dropdown(
            id='x2',
            options=[
                {'label': x, 'value': x} for x in full_options
            ],
            # style={'visibility':'hidden'},
        ),
        dcc.Dropdown(
            id='y2',
            options=[
                {'label': x, 'value': x} for x in full_options
            ],
        ),
        dcc.Dropdown(
            id='z2',
            options=[
                {'label': x, 'value': x} for x in full_options
            ],
        ),

        html.Button(
            'Categorical',
            id='qua1',
            className='qua-button'
        ),

        html.Button(
            'Categorical',
            id='qua2',
            className='qua-button'
        ),

        
    ], className='triple-dropdown'),

    dcc.Graph( 
        className='overview-graph',
        id='graph'
    ),

    html.Div([
        html.Div([
            dt.DataTable(
                data=pp_table.to_dict('records'),
                columns=[{"name": i, "id": i} for i in pp_table.columns],
                row_selectable='single',
                filter_action='native',
                editable=False,
                sort_action='native',
                id='ppTable',
            )
        ], style={'width':'90%', 'margin':'auto', 'top':'50%', 'transform':'translateY(-50%)', 'position':'relative', 'height': '80%', 'overflowY': 'scroll'})
    ], id='ppTableView', style={'display': 'none'}),
    
    html.Div([
        html.Div([
            dt.DataTable(
                data=concentration_table.to_dict('records'),
                columns=[{"name": i, "id": i} for i in concentration_table.columns],
                row_selectable='single',
                filter_action='native',
                editable=False,
                sort_action='native',
                selected_rows=[],
                id='conTable',
            )
        ], style={'width':'90%', 'margin':'auto', 'top':'50%', 'transform':'translateY(-50%)', 'position':'relative', 'height': '80%', 'overflowY': 'scroll'})
    ], id='conTableView', style={'display': 'none'}),

    html.Div([
        html.Div([
            dt.DataTable(
                data=company_table.to_dict('records'),
                columns=[{"name": i, "id": i} for i in company_table.columns],
                row_selectable='single',
                filter_action='native',
                editable=False,
                sort_action='native',
                selected_rows=[],
                id='compTable',
            )
        ], style={'width':'90%', 'margin':'auto', 'top':'50%', 'transform':'translateY(-50%)', 'position':'relative', 'height': '80%', 'overflowY': 'scroll'})
    ], id='compTableView', style={'display': 'none'}),


    html.Button(
        'Counts',
        id='pc',
        className='counts-percent-button'
    ),

    html.Div('Counts', id='pcStore', style={'display':'none'}),
    html.Div('Continuous', id='quaStore1', style={'display':'none'}),
    html.Div('Continuous', id='quaStore2', style={'display':'none'}),

    dcc.Store('oldXYZ', storage_type='memory'),
    dcc.Store(id='current', storage_type='memory'),

    html.Div('', id='selectedQ', style={'display':'none'}),
    html.Div('', id='selectedQ2', style={'display':'none'}),
    html.Div('', id='selectedQ3', style={'display':'none'}),
])

def get_student_to_ratings(assignment, pre_post, rating_type):
    pre_post = 'Pre'

    rating_assignments = assignment_to_ratingIDs[assignment][pre_post]
    
    student_to_ratings = dict()
    
    for assignment in rating_assignments:
        ratings = InnovationRatings.select().where((InnovationRatings.assignment==assignment) & (InnovationRatings.rating_type==rating_type))
        
        for rating in ratings:
            if rating.rating == 'Low':
                rating_number = 1
            elif rating.rating == 'Medium':
                rating_number = 2
            elif rating.rating == 'High':
                rating_number = 3
            
            student_to_ratings[rating.student_id] = rating_number

    return student_to_ratings

def get_student_to_dp(assignment, pre_post):
    pre_post = 'Pre'

    rating_assignments = assignment_to_ratingIDs[assignment][pre_post]
    
    student_to_ratings = dict()
    optionA = None
    optionB = None

    for assignment in rating_assignments:
        responses = QuizResponses.select().where( 
            (QuizResponses.assignment==assignment) & (QuizResponses.question.contains("Decision Point"))
        )
        
        for response in responses:
            if 'Option A' in response.answer:
                rating_number = 1
                if optionA is None:
                    optionA = response.answer
            elif 'Option B' in response.answer:
                if optionB is None:
                    optionB = response.answer
                rating_number = 2

            student_to_ratings[response.student_id] = rating_number
    
    question = response.question.split(": ")[-1]

    return question, student_to_ratings, optionA, optionB

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




def createShowTableCallbacks(match):
    def showTable(selected, x, y, z):
        if not(x == match or y == match or z == match) or selected != []:
            return {'display': 'none'}
        
        return {'position':'absolute', 'top':'20vh', 'height':'80vh', 'width':'100vw', 'backgroundColor':'white'}

    return showTable

def clear_selection(x, y, z):
    return []

for x, y in zip(['ppTable', 'conTable', 'compTable'], ['Personal Survey Questions', 'Concentration', 'Type of Company']):
    app.callback(Output(x + 'View', 'style'), [Input(x, 'selected_rows')], [State('x', 'value'), State('y', 'value'), State('z', 'value')])(createShowTableCallbacks(y))
    app.callback(Output(x, 'selected_rows'), [Input('x', 'value'), Input('y', 'value'), Input('z', 'value')])(clear_selection)




def createStoreSelectedRowCallback(xyz):
    def storeSelectedRow(pp, con, comp, currentXYZ, x,  old):
        if currentXYZ != xyz:
            return old

        selected = None
        if x == 'Personal Survey Questions':
            selected = pp
        elif x == 'Concentration':
            selected = con
        elif x == 'Type of Company':
            selected = comp
            
        if not selected:
            return None

        return selected[0]
    
    return storeSelectedRow

for x, y in zip(['', '2', '3'], ['x', 'y', 'z']):
    app.callback(
        Output('selectedQ' + x, 'children'),
        [Input('ppTable', 'selected_rows'), Input('conTable', 'selected_rows'), Input('compTable', 'selectedRows')], 
        [State('current', 'data'), State(y, 'value'), State('selectedQ' + x, 'children')]
    )(createStoreSelectedRowCallback(y))




@app.callback(Output('oldXYZ', 'data'), [Input('current', 'data')], [State('x', 'value'), State('y', 'value'), State('z', 'value')])
def saveOldXYZ(data, x, y, z):
    return {
        'x': x,
        'y': y,
        'z': z
    }

@app.callback(Output('current', 'data'), [Input('x', 'value'), Input('y', 'value'), Input('z', 'value')], [State('oldXYZ', 'data')])
def saveCurrent(x, y, z, oldXYZ):
    if not oldXYZ:
        return

    for a, b in zip([x, y, z], ['x', 'y', 'z']):
        if oldXYZ[b] != a:
            print('current: ' + b)
            return b




@app.callback(Output('pcStore', 'children'), [Input('pc', 'n_clicks')], [State('pc', 'children')])
def update_pcbutton(n_clicks, quaStore):
    if quaStore == 'Counts':
        return 'Percent'
    else:
        return 'Counts'
    
@app.callback(Output('pc', 'children'), [Input('pcStore', 'children')])
def update_pc(pc):
    return pc


@app.callback(Output('quaStore1', 'children'), [Input('qua1', 'n_clicks')], [State('quaStore1', 'children')])
def update_button(n_clicks, quaStore):
    if quaStore == 'Continuous':
        return 'Categorical'
    else:
        return 'Continuous'
    
@app.callback(Output('qua1', 'children'), [Input('quaStore1', 'children')])
def update_qua(qua):
    return qua

@app.callback(Output('quaStore2', 'children'), [Input('qua2', 'n_clicks')], [State('quaStore2', 'children')])
def update_button2(n_clicks, quaStore):
    if quaStore == 'Continuous':
        return 'Categorical'
    else:
        return 'Continuous'
    
@app.callback(Output('qua2', 'children'), [Input('quaStore2', 'children')])
def update_qua2(qua):
    return qua

def linspace(a, b, n=100):
    if n < 2:
        return b
    diff = (float(b) - a)/(n - 1)
    return [diff * i + a  for i in range(n)]

@app.callback(Output('graph', 'figure'), 
    [
        Input('x', 'value'), Input('x2', 'value'), Input('y', 'value'), 
        Input('y2', 'value'), Input('z', 'value'), Input('z2', 'value'),
        Input('qua1', 'children'), Input('qua2', 'children'), 
        Input('selectedQ', 'children'), Input('selectedQ2', 'children'), Input('selectedQ3', 'children'), Input('pc', 'children')
    ],
    [State('ppTable', 'data'), State('conTable', 'data'), State('compTable', 'data')])
def update_graph(x1, x2, y1, y2, z1, z2, qua1, qua2, question1, question2, question3, pc, pp_questions1, pp_questions2, pp_questions3):
    global db

    db.close()
    db.connect()


    if (x1 and (x2 or question1!='')) and (y1 and (y2 or question2!='')) and (z1 and (z2 or question3!='')):
        dim = [0, 0]
        
        if question1!='':
            if (x1 == 'Personal Survey Questions'):
                q_to_a = question_to_answer
                temp_key = 'Questions'
            elif (x1 == 'Concentration'):
                q_to_a = concentration_to_students
                temp_key = 'Concentrations'
            elif (x1 == 'Type of Company'):
                q_to_a = company_to_students
                temp_key = 'Company Interests'

            student_to_ratings1 = q_to_a[pp_questions1[question1][temp_key]].copy()
            values = list(set(student_to_ratings1.values()))
            values.sort()

            min_value = int(values[0].split('-')[0].strip()) 

            dim[0] = len(values) 
            for key in student_to_ratings1:
                student_to_ratings1[key] = int(student_to_ratings1[key].split('-')[0].strip()) 
                if min_value == 0:
                    student_to_ratings1[key] += 1

            xvals = list(range(min_value-1, dim[0] + min_value-1))
            x = list(range(min_value, dim[0] + min_value)) if qua1=="Continuous" else values
            xprime = list(range(min_value, dim[1] + min_value)) if qua1=="Continuous" else values
            x_title = pp_questions1[question1][temp_key]
        elif 'Ratings' in x2:
            pre_post1 = x2.split('-')[0]
            type1 = x2.split()[0]
            student_to_ratings1 = get_student_to_ratings(x1, pre_post1, type1)
            dim[0] = 3
            x = [0, 1, 2]
            xprime = [1, 2, 3] if qua1=="Continuous" else ["Low", "Medium", "High"]
            x_title = x2 + " for " + x1
        elif 'Decision Point' in x2:
            pre_post1 = x2.split('-')[0]
            question, student_to_ratings1, a, b = get_student_to_dp(x1, pre_post1)
            dim[0] = 2
            x = [0, 1]
            xprime = [1, 2] if qua1=="Continuous" else ['Option A', 'Option B']
            x_title = x2 + " for " + x1

        if question2!='':
            if (y1 == 'Personal Survey Questions'):
                q_to_a = question_to_answer
                temp_key = 'Questions'
            elif (y1 == 'Concentration'):
                q_to_a = concentration_to_students
                temp_key = 'Concentrations'
            elif (y1 == 'Type of Company'):
                q_to_a = company_to_students
                temp_key = 'Company Interests'

            student_to_ratings2 = q_to_a[pp_questions2[question2][temp_key]].copy() 
            values = list(set(student_to_ratings2.values()))
            values.sort()
            min_value = int(values[0].split('-')[0].strip()) 

            dim[1] = len(values) 
            for key in student_to_ratings2:
                student_to_ratings2[key] = int(student_to_ratings2[key].split('-')[0].strip()) 
                if min_value == 0:
                    student_to_ratings2[key] += 1

            y = list(range(min_value-1, dim[1] + min_value-1))
            yprime = list(range(min_value, dim[1] + min_value)) if qua2=="Continuous" else values
            y_title = pp_questions2[question2][temp_key]
        elif 'Ratings' in y2:
            pre_post2 = y2.split('-')[0]
            type2 = y2.split()[0]
            student_to_ratings2 = get_student_to_ratings(y1, pre_post2, type2)
            dim[1] = 3
            y = [0, 1, 2]
            yprime = [1, 2, 3] if qua2=="Continuous" else ["Low", "Medium", "High"]
            y_title = y2 + " for " + y1
        elif 'Decision Point' in y2:
            pre_post2 = y2.split('-')[0]
            question, student_to_ratings2, a, b = get_student_to_dp(y1, pre_post2)
            dim[1] = 2
            y = [0, 1]
            yprime = [1, 2] if qua2=="Continuous" else ['Option A', 'Option B']
            y_title = y2 + " for " + y1
     
        
        if question3!='':
            if (z1 == 'Personal Survey Questions'):
                q_to_a = question_to_answer
                temp_key = 'Questions'
            elif (z1 == 'Concentration'):
                q_to_a = concentration_to_students
                temp_key = 'Concentrations'
            elif (z1 == 'Type of Company'):
                q_to_a = company_to_students
                temp_key = 'Company Interests'

            student_to_ratings3 = q_to_a[pp_questions3[question3][temp_key]].copy() 
            values = list(set(student_to_ratings3.values()))
            values.sort()
            min_value = int(values[0].split('-')[0].strip()) 

            for key in student_to_ratings3:
                student_to_ratings3[key] = int(student_to_ratings3[key].split('-')[0].strip()) 
                if min_value == 0:
                    student_to_ratings3[key] += 1
            titles = values
        elif 'Ratings' in z2:
            pre_post3 = z2.split('-')[0]
            type3 = z2.split('-')[1].split()[1]
            student_to_ratings3 = get_student_to_ratings(z1, pre_post3, type3)
            titles = ["Low", "Medium", "High"]
        elif 'Decision Point' in z2:
            pre_post3 = z2.split('-')[0]
            _, student_to_ratings3, a, b = get_student_to_dp(z1, pre_post3)
            titles = [a, b]

        response_to_students = dict()
        for key, value in student_to_ratings3.items():
            if value not in response_to_students:
                response_to_students[value] = set()
            response_to_students[value].add(key)
  
        keys = sorted(response_to_students.keys())

        if qua1 == 'Continuous' or qua2 == 'Continuous':
            fig = tools.make_subplots(rows=1, cols=1)
            fig['layout'].update(hovermode='closest')
        else:
            fig = tools.make_subplots(rows=1, cols=len(keys), subplot_titles=titles)
        
        maxVal = 0
        all_total = 0
        for index in range(1, len(keys)+1):
            students = response_to_students[keys[index-1]]
            heatmap = [x[:] for x in [[0] * dim[0]] * dim[1]]
            intersect = student_to_ratings1.keys() & student_to_ratings2.keys() & students

            for key in intersect:
                heatmap[student_to_ratings2[key] - 1][student_to_ratings1[key] - 1] += 1

            for i, row in enumerate(heatmap):
                for j, column in enumerate(row):
                    if column > maxVal:
                        maxVal = column
                    all_total += column


        for index in range(1, len(keys)+1):
            students = response_to_students[keys[index-1]]
            heatmap = [x[:] for x in [[0] * dim[0]] * dim[1]]

            intersect = student_to_ratings1.keys() & student_to_ratings2.keys() & students

            for key in intersect:
                heatmap[student_to_ratings2[key] - 1][student_to_ratings1[key] - 1] += 1

           
            sizes = []
            total = 0.0
            for i in range(0, dim[0]):
                for j in range(0, dim[1]):
                    sizes.append(heatmap[j][i])
                    total += heatmap[j][i]
            
          
            row_sums = [0] * dim[1]
            column_sums = [0] * dim[0]

            linreg_x = []
            linreg_y = []
            for i, row in enumerate(heatmap):
                for j, column in enumerate(row):
                    row_sums[i] += column
                    column_sums[j] += column

                    for _ in range(0, column):
                        linreg_x.append(j)
                        linreg_y.append(i)
            
            xprime2 = [0] * len(xprime)
            for i in range(0, len(xprime)):
                xprime2[i] = str(xprime[i]) + "<br />N = " 
                if pc=="Counts":
                    xprime2[i] += str(column_sums[i])
                else:
                    xprime2[i] += str(int(column_sums[i]/all_total*100)) + "%"

            yprime2 = [0] * len(yprime)
            for i in range(0, len(yprime)):
                yprime2[i] = str(yprime[i]) + "<br />N = " 
                if pc=="Counts":
                    yprime2[i] += str(row_sums[i])
                else:
                    yprime2[i] += str(int(row_sums[i]/all_total*100)) + "%"
                
                
          

        
            xvals = []
            for a in x:
                xvals.extend([a] * len(yprime))

            yvals = y*len(xprime)

            # Generated linear fit
            reg_x = linspace(min(linreg_x), max(linreg_x), 20)
            slope, intercept, r_value, p_value, std_err = stats.linregress(linreg_x,  linreg_y)
            slope, intercept, r_value, p_value, std_err = round(slope, 2), round(intercept, 2), round(r_value, 2), round(p_value, 2), round(std_err, 2)
            line = [slope*temp+intercept for temp in reg_x]

            if qua1=='Categorical' and qua2=='Categorical':
                fig.add_trace(go.Heatmap(
                            z=heatmap,
                            x=xprime2,
                            y=yprime2,
                            zauto=False,
                            zmax=maxVal,
                            showscale=True if index==len(keys) else False,
                            colorscale='Blues',
                            reversescale=True
                        ) , 1, index)
            else:
                fig.add_trace(
                    go.Scatter(
                        x=xvals,
                        y=yvals,
                        mode = 'markers+text',
                        marker=dict(
                            size=[x/total*250 for x in sizes],
                            opacity=0.5,
                            color=default_colors[index]
                        ),
                        text = sizes,
                        # hovertext=hover,
                        hoverinfo="text",
                        name=titles[index-1],
                   )
                )

                fig.add_trace(
                    go.Scatter(
                        x=reg_x,
                        y=line,
                        mode='lines',
                        marker=go.Marker(color=default_colors[index]),
                        hovertext="y = {}x + {}<br />r = {}<br />p = {}<br />Standard Error = {}".format(slope, intercept, r_value, p_value, std_err),
                        hoverinfo="text",
                        showlegend=False
                    )
                )

                

                fig.layout = go.Layout(
                    showlegend=True,
                    margin={'l': 225} if question2!='' else {},
                    hovermode='closest',
                    xaxis=dict(
                        title = x_title,
                        zeroline=False,
                        # tickformat = ',d',
                        tickvals=list(range(0, len(x))),
                        ticktext=xprime
                    ),
                    yaxis=dict(
                        title = word_wrap(y_title, 50),
                        zeroline=False,
                        tickvals=list(range(0, len(y) )),
                        ticktext=yprime
                    )
                )
                

 

        return fig 
    if (x1 and (x2 or question1!='')) and (y1 and (y2 or question2!='')):
        dim = [0, 0]
        hover = ""

        if question1!='':
            if (x1 == 'Personal Survey Questions'):
                q_to_a = question_to_answer
                temp_key = 'Questions'
            elif (x1 == 'Concentration'):
                q_to_a = concentration_to_students
                temp_key = 'Concentrations'
            elif (x1 == 'Type of Company'):
                q_to_a = company_to_students
                temp_key = 'Company Interests'

            student_to_ratings1 = q_to_a[pp_questions1[question1][temp_key]].copy()
            values = list(set(student_to_ratings1.values()))
            values.sort()

            min_value = int(values[0].split('-')[0].strip()) 
            min_value = 1 if min_value<1 else min_value

            dim[0] = len(values) 
            for key in student_to_ratings1:
                student_to_ratings1[key] = int(student_to_ratings1[key].split('-')[0].strip()) 
                if min_value == 0:
                    student_to_ratings1[key] += 1

            x = list(range(min_value - 1, dim[0] + min_value - 1))
            xprime = list(range(min_value, dim[0] + min_value)) if qua1=="Continuous" else values
            x_title = pp_questions1[question1][temp_key]
        elif 'Ratings' in x2:
            pre_post1 = x2.split('-')[0]
            type1 = x2.split()[0]
            student_to_ratings1 = get_student_to_ratings(x1, pre_post1, type1)
            dim[0] = 3

            x = [0, 1, 2]
            xprime = [1, 2, 3] if qua1=="Continuous" else ["Low", "Medium", "High"]
            x_title = x2 + " for " + x1
        elif 'Decision Point' in x2:
            pre_post1 = x2.split('-')[0]
            question, student_to_ratings1, a, b = get_student_to_dp(x1, pre_post1)
            dim[0] = 2

            x = [0, 1]
            xprime = [1, 2] if qua1=="Continuous" else ['Option A', 'Option B']
            x_title = x2 + " for " + x1
            hover += question + "<br />" + a + "<br />" + b + "<br /><br />"

        if question2!='':
            if (y1=='Personal Survey Questions'):
                q_to_a = question_to_answer
                temp_key = 'Questions'
            elif (y1=='Concentration'):
                q_to_a = concentration_to_students
                temp_key = 'Concentrations'
            elif (y1=='Type of Company'):
                q_to_a = company_to_students
                temp_key = 'Company Interests'

            student_to_ratings2 = q_to_a[pp_questions2[question2][temp_key]].copy() 
            values = list(set(student_to_ratings2.values()))
            values.sort()
            min_value = int(values[0].split('-')[0].strip()) 

            dim[1] = len(values) 
            for key in student_to_ratings2:
                student_to_ratings2[key] = int(student_to_ratings2[key].split('-')[0].strip()) 
                if min_value == 0:
                    student_to_ratings2[key] += 1

            y = list(range(min_value - 1, dim[1] + min_value - 1)) 
            yprime = list(range(min_value, dim[1] + min_value)) if qua2=="Continuous" else values
            y_title = pp_questions2[question2][temp_key]
        elif 'Ratings' in y2:
            pre_post2 = y2.split('-')[0]
            type2 = y2.split()[0]
            student_to_ratings2 = get_student_to_ratings(y1, pre_post2, type2)
            dim[1] = 3

            y = [0, 1, 2]
            yprime = [1, 2, 3] if qua2=="Continuous" else ["Low", "Medium", "High"]
            y_title = y2 + " for " + y1
        elif 'Decision Point' in y2:
            pre_post2 = y2.split('-')[0]
            question, student_to_ratings2, a, b = get_student_to_dp(y1, pre_post2)
            dim[1] = 2

            y = [0, 1]
            yprime = [1, 2] if qua2=="Continuous" else ['Option A', 'Option B']
            y_title = y2 + " for " + y1
            hover += question + "<br />" + a + "<br />" + b
        
        intersect = student_to_ratings1.keys() & student_to_ratings2.keys()
        heatmap = [x[:] for x in [[0] * dim[0]] * dim[1]]

        for key in intersect:
            heatmap[student_to_ratings2[key] - 1][student_to_ratings1[key] - 1] += 1

        if qua1 == 'Continuous' or qua2 == 'Continuous':
            sizes = []
            for i in range(0, dim[0]):
                for j in range(0, dim[1]):
                    sizes.append(heatmap[j][i])
         
    
        row_sums = [0] * dim[1]
        column_sums = [0] * dim[0]

        linreg_x = []
        linreg_y = []

        total = 0
        for i, row in enumerate(heatmap):
            for j, column in enumerate(row):
                row_sums[i] += column
                column_sums[j] += column
                total += column

                for _ in range(0, column):
                    linreg_x.append(j)
                    linreg_y.append(i)
        
        for i in range(0, len(xprime)):
            if pc=='Counts':
                xprime[i] = str(xprime[i]) + "<br />" + "N = " + str(column_sums[i])
            else:
                xprime[i] = str(xprime[i]) + "<br />" + "N = " + str(int(column_sums[i]/total*100)) + "%"

        for i in range(0, len(yprime)):
            if pc=='Counts':
                yprime[i] = str(yprime[i]) + "<br />" + "N = " + str(row_sums[i])
            else:
                yprime[i] = str(yprime[i]) + "<br />" + "N = " + str(int(row_sums[i]/total*100)) + "%"

           

        xvals = []
        for a in x:
            xvals.extend([a] * len(yprime))

        yvals = y*len(xprime)


        # Generated linear fit
        reg_x = linspace(min(linreg_x), max(linreg_x), 20)
        slope, intercept, r_value, p_value, std_err = stats.linregress(linreg_x,  linreg_y)
        slope, intercept, r_value, p_value, std_err = round(slope, 2), round(intercept, 2), round(r_value, 2), round(p_value, 2), round(std_err, 2)
        line = [slope*temp+intercept for temp in reg_x]

        data = []

        heatmap_text = [[hover] * dim[0] for i in range(dim[1])]
        for i in range(0, dim[1]):
            for j in range(0, dim[0]):
                heatmap_text[i][j] = '{} ({}%)<br />{}'.format(heatmap[i][j], round(100.0*heatmap[i][j]/total, 2), heatmap_text[i][j])
             
                    

        if qua1 == 'Categorical' and qua2 == 'Categorical':
            data = [
                go.Heatmap(
                    z=heatmap,
                    x=xprime,
                    y=yprime,
                    colorscale='Blues',
                    reversescale=True,
                    text=heatmap_text,
                    hoverinfo="text",
                )
            ]
        else:
            data = [
                go.Scatter(
                    x=xvals,
                    y=yvals,

                    mode = 'markers+text',
                    marker=dict(
                        size=[x/total*250 for x in sizes],
                    ),
                    text = sizes,
                    hovertext=hover,
                    hoverinfo="text",
                    
                ),
                go.Scatter(
                    x=reg_x,
                    y=line,
                    mode='lines',
                    marker=go.Marker(color='rgb(31, 119, 180)'),
                    hovertext="y = {}x + {}<br />r = {}<br />p = {}<br />Standard Error = {}".format(slope, intercept, r_value, p_value, std_err),
                    hoverinfo="text",
                )
            
            ]

        return go.Figure(
            data=data,
            layout = go.Layout(
                showlegend=False,
                margin={'l': 225} if question2!='' else {},
                hovermode='closest',
                xaxis=dict(
                    title = x_title,
                    zeroline=False,
                    # tickformat = ',d',
                    tickvals=list(range(0, len(x))),
                    ticktext=xprime
                ),
                yaxis=dict(
                    title = word_wrap(y_title, 50),
                    zeroline=False,
                    tickvals=list(range(0, len(y) )),
                    ticktext=yprime
                )
            )
        )
    if (x1 and (x2 or question1!='')) or  (y1 and (y2 or question2!='')):
        if (x1 and x2 and 'Ratings' in x2) or (y1 and y2 and 'Ratings' in  y2):
            if x1 and x2:
                pre_post = 'Pre'
                rt = x2.split()[0]
                student_to_ratings = get_student_to_ratings(x1, pre_post, rt)
                counts = dict(Counter(student_to_ratings.values()))
                xlabel = x2 + " for " + x1
                bar = True if qua1 == "Categorical" else False
                x = ['Low', 'Medium', 'High'] if qua1 == 'Categorical' else [1, 2, 3]
                if pc=='Counts':
                    y = [counts[1], counts[2], counts[3]]
                    ylabel="Number of Ratings"
                else:
                    total = sum([counts[1], counts[2], counts[3]])
                    y = [counts[1]/total*100, counts[2]/total*100, counts[3]/total*100]
                    ylabel='Percent (%)'

            elif y1 and y2:
                pre_post = 'Pre'
                rt = y2.split()[0]
                student_to_ratings = get_student_to_ratings(y1, pre_post, rt)
                counts = dict(Counter(student_to_ratings.values()))
                ylabel = y2 + " for " + y1
                bar = True if qua2 == "Categorical" else False
                if pc=='Counts':
                    x = [counts[1], counts[2], counts[3]] 
                    xlabel="Number of Ratings"
                else:
                    total = sum([counts[1], counts[2], counts[3]])
                    x = [counts[1]/total*100, counts[2]/total*100, counts[3]/total*100]
                    xlabel='Percent (%)'
                
                y = ['Low', 'Medium', 'High'] if qua2 == 'Categorical' else [1, 2, 3]

            return go.Figure(
                data=[
                    go.Bar(
                        x=x,
                        y=y,
                        marker=dict(color=colors),
                        orientation='h' if (y1 and y2) else 'v'
                    ) if bar else
                    go.Scatter(
                        x=x,
                        y=y,
        
                        mode = 'markers',

                        marker=dict(
                            size=16,
                            color=colors,
                        )
                    )
                ],
                layout = go.Layout(
                    xaxis=dict(
                        title = xlabel,
                        tickformat = ',d',
                        range=[0, 100] if xlabel=='Percent (%)' else None
                    ),
                    yaxis=dict(
                        title = ylabel,
                        tickformat = ',d',
                        rangemode = "tozero",
                        range=[0, 100] if xlabel=='Percent (%)' else None
                    )
                )
            )
        elif x2 == 'Decision Point' or y2 == 'Decision Point':
            if x1 and x2:
                pre_post = 'Pre'
                question, student_to_dp, a, b = get_student_to_dp(x1, pre_post)
                counts = dict(Counter(student_to_dp.values()))

                if counts is None:
                    return go.Figure()

                keys = list(dict(counts).keys())
                keys.sort()

                bar = True if qua1 == "Categorical" else False
                x = ['Option A', 'Option B'] if qua1 == 'Categorical' else [1, 2]
                y = [counts[x] for x in keys]

                xlabel='Decision'
                ylabel='Counts'
                
                if pc=='Percent':
                    total = sum(y)
                    y = [a/total*100 for a in y]
                    ylabel='Percents (%)'

            elif y1 and y2:
                pre_post = 'Pre'
                question, student_to_dp, a, b = get_student_to_dp(y1, pre_post)
                counts = dict(Counter(student_to_dp.values()))

                if counts is None:
                    return go.Figure()

                keys = list(dict(counts).keys())
                keys.sort()

                bar = True if qua2 == "Categorical" else False
                x = [counts[x] for x in keys]
                y = ['Option A', 'Option B'] if qua2 == 'Categorical' else [1, 2]

                xlabel='Counts'
                ylabel='Decision'
                
                if pc=='Percent':
                    total = sum(x)
                    x = [a/total*100 for a in x]
                    xlabel='Percents (%)'

        
            return go.Figure(
                data=[
                    go.Bar(
                        x=x,
                        y=y,
                        text = [word_wrap(a, 50) , word_wrap(b, 50)],
                        marker=dict(color=colors),
                        orientation='h' if (y1 and y2) else 'v'
                    ) if bar else
                    go.Scatter(
                        x=x,
                        y=y,
                        text = [word_wrap(a, 50) , word_wrap(b, 50)],
                        mode = 'markers',

                        marker=dict(
                            size=16,
                            color=colors,
                        )
                    )
                ],
                layout = go.Layout(
                    title=question,
                    xaxis=dict(
                        title=xlabel,
                        tickformat = ',d',
                        range=[0, 100] if xlabel=='Percent (%)' else None
                    ),
                    yaxis=dict(
                        title=ylabel,
                        rangemode = "tozero",
                        tickformat = ',d',
                        range=[0, 100] if xlabel=='Percent (%)' else None
                    )
                )
            )
        else:
            if (x1 == 'Personal Survey Questions' or y1=='Personal Survey Questions'):
                q_to_a = question_to_answer
                key = 'Questions'
            elif (x1 == 'Concentration' or y1=='Concentration'):
                q_to_a = concentration_to_students
                key = 'Concentrations'
            elif (x1 == 'Type of Company' or y1=='Type of Company'):
                q_to_a = company_to_students
                key = 'Company Interests'

            if question1 != '':
                counts = dict(Counter(q_to_a[pp_questions1[question1][key]].values()))
                title = pp_questions1[question1][key]
                orein = 'v'
                keys = list(counts.keys())
                keys.sort()
                x = keys if qua1=="Categorical" else list(range(1, len(keys) + 1))
                y = [counts[x] for x in keys]
                
                xlabel='Response'
                ylabel='Counts'

                if pc=='Percent':
                    total = sum(y)
                    y = [a/total*100 for a in y]
                    ylabel='Percents (%)'
            else:
                counts = dict(Counter(q_to_a[pp_questions2[question2][key]].values()))
                title = pp_questions2[question2][key]
                orein = 'h'
                keys = list(counts.keys())
                keys.sort()
                y = keys if qua2=="Categorical" else list(range(1, len(keys) + 1))
                x = [counts[x] for x in keys]

                ylabel='Response'
                xlabel='Counts'

                if pc=='Percent':
                    total = sum(x)
                    x = [a/total*100 for a in x]
                    xlabel='Percents (%)'
            
            return go.Figure(
                data=[
                    go.Bar(
                        x=x,
                        y=y,
                        orientation=orein
                    ) 
                ],
                layout = go.Layout(
                    title=title,
                    margin={'l': 200} if orein=='h' else {},
                    xaxis=dict(
                        title=xlabel,
                        tickformat = ',d',
                        range=[0, 100] if xlabel=='Percent (%)' else None
                    ),
                    yaxis=dict(
                        title=ylabel, 
                        tickformat = ',d',
                        rangemode = "tozero",
                        range=[0, 100] if xlabel=='Percent (%)' else None
                    )
                )
            )
        
    return go.Figure()

@app.callback(Output('x2', 'options'), [Input('x', 'value'), Input('y2', 'value'), Input('y', 'value')])
def update_x2(x1, y2, y1):
    if x1 is None:
        return []
    else:
        return [{'label': x, 'value': x} for x in full_options]

@app.callback(Output('y2', 'options'), [Input('y', 'value'), Input('x2', 'value'), Input('x', 'value')])
def update_y2(y1, x2, x1):
    if y1 is None:
        return []
    else:
        return [{'label': x, 'value': x} for x in full_options]

@app.callback(Output('z2', 'options'), [Input('z', 'value')])
def update_z2(z1):
    if z1 is None:
        return []
    else:
        return [{'label': x, 'value': x} for x in full_options]

@app.callback(Output('x2', 'disabled'), [Input('x', 'value')])
def disablex2(x1):
    if x1 == "Personal Survey Questions" or x1=='Concentration' or x1=='Type of Company':
        return True
    return False

@app.callback(Output('x2', 'value'), [Input('x', 'value')], [State('x2', 'value')])
def clear_x2(x1, x2):
    if x1=='Personal Survey Questions' or x1=='Concentration' or x1=='Type of Company':
        return None
    return x2

@app.callback(Output('y2', 'disabled'), [Input('y', 'value')])
def disabley2(y1):
    if y1 == "Personal Survey Questions" or y1=='Concentration' or y1=='Type of Company':
        return True
    return False

@app.callback(Output('y2', 'value'), [Input('y', 'value')], [State('y2', 'value')])
def clear_y2(y1, y2):
    if y1=='Personal Survey Questions' or y1=='Concentration' or y1=='Type of Company':
        return None
    return y2

@app.callback(Output('z2', 'disabled'), [Input('z', 'value')])
def disablez2(z1):
    if z1 == "Personal Survey Questions" or z1=='Concentration' or z1=='Type of Company':
        return True
    return False

@app.callback(Output('z2', 'value'), [Input('z', 'value')], [State('z2', 'value')])
def clear_z2(z1, z2):
    if z1=='Personal Survey Questions' or z1=='Concentration' or z1=='Type of Company':
        return None
    return z2
