from django.shortcuts import render

from .models import *

from rest_framework.response import Response
from rest_framework.decorators import api_view

import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY", None)

@api_view(['GET', 'POST'])
def chat(request):
    chatbot_response = None

    if openai.api_key is not None and request.method == 'POST':
        user_input = request.data.get('user_input')  # Use request.data for POST data
        username = request.data.get('username')  # Use request.data for username

        current_patient = Patient.objects.get(user__username=username)

        medical_history = " * ".join(history.description for history in current_patient.medical_history.all())
        prompt_history = " * ".join(prompt.description for prompt in current_patient.prompt_history.all())

        prompt = f'Patient history: {medical_history}\nPatient prompts: {prompt_history}\nAge: {current_patient.age}\nSex: {current_patient.gender}\nHeight: {current_patient.height}\nWeight: {current_patient.weight}\nUser input: {user_input}'

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "Assist as a medical AI. Respond briefly to the user input, considering the prompt and medical history"},
                {"role": "user", "content": prompt}
            ],
            #max_tokens=100
        )   
        prompt_to_db = Prompt.objects.create(description=user_input) 
        current_patient.prompt_history.add(prompt_to_db)
        current_patient.save()

        chatbot_response = response.choices[0].message['content']

    return Response({'response': chatbot_response})

@api_view(['GET', 'POST'])
def login(request):
    user=User.objects.get(username=request.data.get("username"))

