from django.shortcuts import render

from django.http import JsonResponse

import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY", None)


from django.http import JsonResponse

def chat(request):
    chatbot_response = None

    if openai.api_key is not None and request.method == 'GET':
        user_input = request.GET.get('user_input')
        prompt = f'Act like a medical assistant or doctor and answer {user_input}'

        response = openai.Completion.create(
            engine="text-davinci-003",  # You can use "text-davinci-003" for GPT-3.5 Turbo
            prompt=prompt,
            max_tokens=200  # Adjust the desired response length
        )
        
        chatbot_response = response.choices[0].text.strip()

    print(chatbot_response)  # This will print the chatbot response in your terminal

    return JsonResponse({'response': chatbot_response})

