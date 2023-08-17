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
    ('Non-binary', 'Non-binary'),
)


class Patient(models.Model):
    user = models.ForeignKey(User, related_name='profile', on_delete=models.CASCADE)
    medical_history = models.ManyToManyField(History, related_name='patient_with_history')
    prompt_history = models.ManyToManyField(Prompt, related_name='patient_with_prompt')
    age = models.IntegerField()
    gender = models.CharField(max_length=15,choices=Sex)
    height = models.DecimalField(max_digits=5, decimal_places=2)
    weight = models.DecimalField(max_digits=5, decimal_places=2)

    def __str__(self):
        return self.user.username
