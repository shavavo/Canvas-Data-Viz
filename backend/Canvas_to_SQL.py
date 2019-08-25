import peewee
import sqlCredentials
from peewee import *
from canvasapi import Canvas
import canvasCredentials
import requests
import json
import pandas as pd
import os
import time
from pandas.compat import StringIO
import numpy as np
from datetime import datetime
import dateutil
from dateutil.tz import tzutc

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
import sys

import canvasModels
from canvasModels import *
db = canvasModels.init_db(sqlCredentials)
canvas = Canvas(canvasCredentials.API_URL, canvasCredentials.API_KEY)

sys.stdout = open("logs.txt", "w")

last_update = None


def populateSectionsAndStudents(course_num):
    course = canvas.get_course(course_num)
    groups = list(course.get_sections())

    for group in groups:
        try:
            section_number = int(group.name[-2] + group.name[-1])
        except ValueError:
            # Not a section we care about, exm: 'Other'
            continue

        try:
            section = Sections.get_by_id(group.id)
        except peewee.DoesNotExist:
            section = Sections.create(canvas_id=group.id, name=group.name, course_id=group.course_id, section_number=section_number)

        enrollments = group.get_enrollments()
        
        for enrollment in enrollments:
            if(enrollment.type == "StudentEnrollment"):
                try:
                    student = Students.get_by_id(enrollment.user_id)
                except peewee.DoesNotExist:
                    student = Students.create(canvas_id=enrollment.user_id, name=enrollment.user['name'], raw=json.dumps(enrollment.user))

                StudentSections.get_or_create(student=student, section=section)

def populateAssignments(course_num):
    with db.atomic():
        course = canvas.get_course(course_num)
        assignments = course.get_assignments()

        to_insert = []
        for assignment in assignments:
            try:
                name = assignment.name
                section_number = int(name[name.find("(Sec. ")+6:name.find(")")])
            except ValueError:
                section_number = 1

            try:
                Assignments.get_by_id(assignment.id)
            except peewee.DoesNotExist:
                section = Sections.get(Sections.course_id==course.id, Sections.section_number==section_number)
                to_insert.append(
                    {'assignment_id': assignment.id, 'name': assignment.name, 'raw': json.dumps(assignment.attributes), 'section': section}
                )
        if(len(to_insert) != 0):
            Assignments.insert_many(to_insert).execute()

def downloadQuizResponses(course_num):
    global last_update

    course = canvas.get_course(course_num)

    assignments = course.get_assignments()

    for assignment in assignments:
        if('online_quiz' in assignment.submission_types):
            post = requests.post(canvasCredentials.API_URL + "/api/v1/courses/" + str(course_num) + "/quizzes/" + str(assignment.quiz_id) + "/reports" + "?access_token=" + canvasCredentials.API_KEY,
                                    data='quiz_report[report_type]=student_analysis')
            
            post_dict = json.loads(post.content)
            
            d = dateutil.parser.parse(post_dict['updated_at'])

            if last_update is not None and d < last_update:
                continue
            
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
            file_name = str(assignment.id) + '.csv'
            
            download = requests.get(download_link)

            TESTDATA = StringIO(download.content.decode("utf-8") )
            data = pd.read_csv(TESTDATA)
            data.to_csv('assignments/' + file_name, encoding='utf-8')
        else:
            pass



def populateQuizReponses():
    # enrollment = pd.read_excel('mgmt738_enrolled.xlsx')
    # enrollment = enrollment.replace(101, 'MANAGEMT 738.101').replace(102, 'MANAGEMT 738.102').replace(103, 'MANAGEMT 738.103').replace(104, 'MANAGEMT 738E.601')

    with db.atomic():
        csv_files = []
        names = []

        for filename in os.listdir('assignments/'):
            if filename.endswith(".csv"): 
                data = pd.read_csv("assignments/" + filename, index_col=0, encoding='utf-8')
                csv_files.append(data)
                names.append(filename.replace('.csv', ''))

        print('Updating ' + str(len(csv_files)) + ' assignment(s)')

        for i, data in enumerate(csv_files):
            to_insert = []

            try:
                assignment = Assignments.get_by_id(names[i])
            except peewee.DoesNotExist:
                populateAssignments(801)
                populateAssignments(812)
                assignment = Assignments.get_by_id(names[i])
          
            for j, row in data.iterrows():
                for k in range(7, data.shape[1]-3, 2):
                    student = Students.get_by_id(row.id)
                    
                    question = row.index[k] 
                    answer = row[k] 

                    if answer is np.NaN:
                        continue
                    
                    assigned_id = int(question[0:5] + names[i] + str(row.id))
                                    
                    student_sections = [int(x) for x in str(row['section_id']).split(', ')]
                    if assignment.section.canvas_id not in student_sections:
                        break

                    try:
                        qr = QuizResponses.get_by_id(assigned_id)
                        if qr.answer != answer:
                            qr.answer = answer
                            qr.save()
                    except peewee.DoesNotExist:
                        student = Students.get_by_id(row.id)
                        assignment = Assignments.get_by_id(names[i])

                        to_insert.append({
                            'assigned_id': assigned_id, 
                            'question': question, 
                            'answer': answer, 
                            'assignment': assignment, 
                            'student': student,
                            'section': assignment.section
                        })
                    
                    if 'Your rating' in question:
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
                            continue

                        if "pre-discussion" in assignment.name:
                            pre_post = "Pre"
                        elif "post-discussion" in assignment.name:
                            pre_post= "Post"
                        else:
                            continue
                    elif 'Evidence in support' in question:
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
                        assigned_id += str(names[i])
                        assigned_id = int(assigned_id)

                        if answer == '1-Low' or answer == '2-Medium' or answer == '3-High':
                            answer = ''
                    
                        try:
                            InnovationRatings.get_by_id(assigned_id)
                        except peewee.DoesNotExist:
                            InnovationRatings.create(assigned_id=assigned_id, pre_or_post=pre_post, rating_type=rating_type, rating=rating, justification=answer, assignment=assignment, student=student, section=assignment.section)


                        
            if len(to_insert) != 0:
                QuizResponses.insert_many(to_insert).execute()

def populateDecisionPoints():
    with db.atomic():
        companies = ['IKEA', 'Iridium', 'athenahealth', 'BMW', 'Google', 'P&G', 'IBM', 'Unilever', 'Venmo', '3M', 'Airbnb']
        pre_rating_assignments = {}
        post_rating_assignments = {}
        writeup_assignments = {}

        sections = Sections.select()
        for section in sections:
            pre_rating_assignments[section] = []
            post_rating_assignments[section] = []
            writeup_assignments[section] = []
            
            for company in companies:
                assignments = Assignments.select().where( (Assignments.name.contains(company)) & (Assignments.section==section))
                
                if( len(assignments)==2 ):
                    writeup_assignments[section].append(None)
                
                for assignment in assignments:
                    if('pre-discussion' in assignment.name):
                        pre_rating_assignments[section].append(assignment)
                    elif('post-discussion' in assignment.name):
                        post_rating_assignments[section].append(assignment)
                    elif('Write-up' in assignment.name):
                        writeup_assignments[section].append(assignment)

        for section in sections:
            to_insert = []
            
            for assignment in post_rating_assignments[section]:
                post_decision_points = QuizResponses.select().where( (QuizResponses.assignment==assignment) & (QuizResponses.question.contains('Decision Point')) )

                for dp in post_decision_points:
                    assigned_id = str(dp.assignment_id) + str(dp.student_id)

                    try:
                        DecisionPoints.get_by_id(assigned_id)
                    except peewee.DoesNotExist:
                        to_insert.append(
                            {'assigned_id': assigned_id, 
                            'pre_or_post': 'Post', 
                            'decision': dp.answer, 
                            'justification': "", 
                            'student': dp.student, 
                            'section': section, 
                            'assignment': assignment}
                        )
                    
            for index, assignment in enumerate(pre_rating_assignments[section]):
                pre_decision_points = QuizResponses.select().where( (QuizResponses.assignment==assignment) & (QuizResponses.question.contains('Decision Point')) )

                for dp in pre_decision_points:
                    choice = '(a)' if('Option A' in dp.answer) else '(b)'

                    writeup_assignment = writeup_assignments[section][index]
                    justification = QuizResponses.select().where( (QuizResponses.assignment==writeup_assignment) & (QuizResponses.student==dp.student) & (QuizResponses.question.contains(choice)) ).first()

                    justification = "" if(justification is None) else justification.answer

                    assigned_id = str(dp.assignment_id) + str(dp.student_id)

                    try:
                        DecisionPoints.get_by_id(assigned_id)
                    except peewee.DoesNotExist:
                        to_insert.append(
                            {'assigned_id': assigned_id, 
                            'pre_or_post': 'Pre', 
                            'decision': dp.answer, 
                            'justification': justification, 
                            'student': dp.student, 
                            'section': section, 
                            'assignment': assignment}
                        )
                        
            if(len(to_insert) != 0):
                DecisionPoints.insert_many(to_insert).execute()
        

def isValid(x):
    try:
        y = float(x)
    except:
        return True
    return False

def clear_assignments_folder():
    for filename in os.listdir('assignments/'):
        if filename.endswith(".csv"): 
            os.remove("assignments/" + filename)

def update():
    global last_update

    next_last_update = datetime.utcnow().replace(tzinfo=tzutc())

    print(next_last_update.strftime('%Y-%m-%d %H:%M'))
    
    db.close()
    db.connect()

    db.create_tables([Sections, Students, StudentSections, Assignments, QuizResponses, InnovationRatings])
        
    # print('Populating Students, Sections..')
    # populateSectionsAndStudents(801)
    # populateSectionsAndStudents(812)

    # print('Populating Assignments..')
    # populateAssignments(801)
    # populateAssignments(812)

    downloadQuizResponses(801)
    downloadQuizResponses(812)

    populateQuizReponses()
    populateDecisionPoints()
    
    print("Completed")
    db.close()

    clear_assignments_folder()
    last_update = next_last_update


def main():
    text_file = open("save_pid.txt", "w")
    text_file.write(str(os.getpid()))
    text_file.close()

    update()
    scheduler = BlockingScheduler(executors={'default': ThreadPoolExecutor(1)})
    scheduler.add_job(update, 'interval', hours=1)
    # scheduler.add_job(update, 'cron', day_of_week="mon,thu", hour="9-18", minute="*")
    
    scheduler.start()


    scheduler.start()


if __name__ == "__main__":
    main()