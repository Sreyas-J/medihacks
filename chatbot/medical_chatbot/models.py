from django.db import models
from django.contrib.auth.models import User

class History(models.Model):
    description = models.TextField()
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.description
    
class Prompt(models.Model):
    description = models.TextField()
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.description

Sex = (
    ('Male', 'Male'),
    ('Female', 'Female'),
)

class Report(models.Model):
    HighBP=models.BooleanField()
    HighChol=models.BooleanField()
    GenHlth=models.IntegerField()   #no user input
    CholCheck=models.BinaryField()
    BMI=models.DecimalField(max_digits=6,decimal_places=2)  #no user input
    Smoker=models.BooleanField()
    Stroke=models.BooleanField()
    HeartDiseaseorAttack=models.BooleanField()
    

class Patient(models.Model):
    user = models.ForeignKey(User, related_name='profile', on_delete=models.CASCADE)
    medical_history = models.ManyToManyField(History, related_name='patient_with_history')
    prompt_history = models.ManyToManyField(Prompt, related_name='patient_with_prompt')
    health_report=models.ForeignKey(Report,related_name='report_card',on_delete=models.SET_NULL,null=True)

    BP = models.DecimalField(max_digits=6, decimal_places=2,null=True)
    Cholestrol= models.DecimalField(max_digits=7,decimal_places=2,null=True)
    Cholestrol_check=models.BooleanField(null=True)
    Smoker=models.BooleanField(null=True)
    Stroke=models.BooleanField(null=True)
    HeartDisease=models.BooleanField(null=True)

    def __str__(self):
        return self.user.username
