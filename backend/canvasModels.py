import peewee
from peewee import *
import sqlCredentials

db_proxy = Proxy()

class BaseModel(Model):
    class Meta:
        database = db_proxy

class Sections(BaseModel):
    canvas_id = IntegerField(primary_key=True, column_name='section_id')
    name = TextField()
    course_id = IntegerField()
    section_number = IntegerField()

class Students(BaseModel):
    canvas_id = IntegerField(primary_key=True)
    name = TextField()
    raw = TextField()

class StudentSections(BaseModel):
    student = ForeignKeyField(Students)
    section = ForeignKeyField(Sections)
    

class Assignments(BaseModel):
    assignment_id = IntegerField(primary_key=True)
    name = TextField()
    raw = TextField()
    section = ForeignKeyField(Sections, backref="assignments")

class QuizResponses(BaseModel):
    assigned_id = BigIntegerField(primary_key=True)

    question = TextField()
    answer = TextField()
    assignment = ForeignKeyField(Assignments, backref="responses")
    student = ForeignKeyField(Students, backref="response")
    section = ForeignKeyField(Sections, backref="response")

class InnovationRatings(BaseModel):
    assigned_id = BigIntegerField(primary_key=True)

    pre_or_post = TextField()
    rating_type = TextField()
    rating = TextField()
    justification = TextField()
    
    assignment = ForeignKeyField(Assignments, backref="responses")
    student = ForeignKeyField(Students, backref="response")
    section = ForeignKeyField(Sections, backref="response")

        
class DecisionPoints(BaseModel):
    assigned_id = BigIntegerField(primary_key=True)
    pre_or_post = TextField()
    
    decision = TextField()
    justification = TextField()
    
    assignment = ForeignKeyField(Assignments, backref="responses")
    student = ForeignKeyField(Students, backref="response")
    section = ForeignKeyField(Sections, backref="response")

def init_db(sqlCredentials):
    db = MySQLDatabase(sqlCredentials.name, host=sqlCredentials.host, port=sqlCredentials.port, user=sqlCredentials.user, passwd=sqlCredentials.password )
    db_proxy.initialize(db)

    return db