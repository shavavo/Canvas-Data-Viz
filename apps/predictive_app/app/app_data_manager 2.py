from canvasModels import *
import canvasModels
db = canvasModels.init_db(sqlCredentials)
import pandas as pd
from collections import Counter

def refresh_connection():
    db.close()
    db.connect()

assignment_names = ['IKEA', 'Iridium', 'athenahealth', 'BMW', 'Google', 'P&G', 'IBM', 'Unilever', 'Venmo', '3M', 'Airbnb', 'Personal Survey Questions', 'Concentration', 'Type of Company']
assignments = Assignments.select()
assignment_to_ratingIDs = dict()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for assignment in assignments:
    if 'innovation ratings' in assignment.name:
        company = assignment.name.split("pre-discussion")[0].split("post-discussion")[0].strip().split()[-1]
        pre_post = 'pre' if 'pre-discussion' in assignment.name else 'post'
        
        if company not in assignment_to_ratingIDs:
            assignment_to_ratingIDs[company] = dict()
        if pre_post not in assignment_to_ratingIDs[company]:
            assignment_to_ratingIDs[company][pre_post] = []
            
        assignment_to_ratingIDs[company][pre_post].append(assignment)



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

additional_questions = [
    ["What type of company are you interested in studying for the final project in this course (check all that apply)?", companies],
    ["What is your current business school concentration (check all that apply)?", concentrations]
]

def binarize_question(question_to_answer, all_answers, question):
    refresh_connection()

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
    

def get_student_to_ratings(assignment, pre_post, rating_type):
    refresh_connection()

    rating_assignments = assignment_to_ratingIDs[assignment][pre_post]
    
    student_to_ratings = dict()
    
    for assignment in rating_assignments:
        ratings = InnovationRatings.select().where((InnovationRatings.assignment==assignment) & (InnovationRatings.rating_type==rating_type))
        
        for rating in ratings:
            if rating.rating == 'Low':
                rating_number = '1 - Low'
            elif rating.rating == 'Medium':
                rating_number = '2 - Medium'
            elif rating.rating == 'High':
                rating_number = '3 - High'
            
            student_to_ratings[rating.student_id] = rating_number

    return student_to_ratings

def get_student_to_dp(assignment, pre_post):
    refresh_connection()

    rating_assignments = assignment_to_ratingIDs[assignment][pre_post]
    
    student_to_ratings = dict()
    optionA = None
    optionB = None

    response = None

    for assignment in rating_assignments:
        responses = QuizResponses.select().where( 
            (QuizResponses.assignment==assignment) & (QuizResponses.question.contains("Decision Point"))
        )
        
        for response in responses:
            if 'Option A' in response.answer:
                rating_number = '1 - ' + response.answer.replace(' - ', ': ')
                if optionA is None:
                    optionA = response.answer
            elif 'Option B' in response.answer:
                if optionB is None:
                    optionB = response.answer
                rating_number = '2 - ' + response.answer.replace(' - ', ': ')

            student_to_ratings[response.student_id] = rating_number
    
    if response:
        question = response.question.split(": ")[-1]

    return question, student_to_ratings, optionA, optionB

def get_pp():
    refresh_connection()
    
    pp_assignments = list(Assignments.select().where(Assignments.name.contains("Personal Preference Survey")))
    pp_questions = []

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
    
    

    for x in additional_questions:
        answer_to_students = binarize_question(question_to_answer, x[1], x[0])
        # pp_questions.append(x[0])
        question_to_answer[x[0]] = answer_to_students

    # concentration_table = pd.DataFrame(list(concentration_to_students.keys()))
    # concentration_table.columns = ['Concentrations']
    # concentration_dict = concentration_table.to_dict('records')

    # company_table = pd.DataFrame(list(company_to_students.keys()))
    # company_table.columns = ['Company Interests']
    # company_dict = company_table.to_dict('records')

    # print(concentration_to_students)
 
    return question_to_answer, pp_questions



    
