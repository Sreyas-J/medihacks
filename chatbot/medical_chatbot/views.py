from django.shortcuts import render

from .models import *

from django.http import JsonResponse

import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY", None)

def chat(request):
    chatbot_response = None

    if openai.api_key is not None and request.method == 'GET':
        user_input = request.GET.get('user_input')

        current_patient = Patient.objects.get(id=1) 
        medical_history = " * ".join(history.description for history in current_patient.medical_history.all())
        prompt_history = " * ".join(prompt.description for prompt in current_patient.prompt_history.all())

        prompt = f'Patient history: {medical_history}\nPatient prompts: {prompt_history}\nAge: {current_patient.age}\nSex: {current_patient.gender}\nHeight: {current_patient.height}\nWeight: {current_patient.weight}\nUser input: {user_input}'
        print(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "Try to be a medical assistant and answer the user input."},
                {"role": "user", "content": prompt}
            ],
            #max_tokens=100
        )   
        prompt_to_db=Prompt.objects.create(description=user_input) 
        current_patient.prompt_history.add(prompt_to_db)
        current_patient.save()

        chatbot_response = response.choices[0].message['content']

    return JsonResponse({'response': chatbot_response})



