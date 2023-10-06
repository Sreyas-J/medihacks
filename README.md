
# Medical Chatbot

- Backend for a chatbot that can provide basic context-based medical suggestions. 
- The code also has a component for diabetes prediction that can be integrated with the backend to provide an API endpoint for predictions.

## Tech Stack


**Backend:** Django Rest FrameWork

**ML Algorithm:** TensorFlow


## API Reference

The chatbot is implemented using OPENAI's API.


## Run Locally

Clone the project

```bash
  git clone git@github.com:Sreyas-J/medihacks.git
```

Go to the project directory

```bash
  cd medihacks
```

Install dependencies

```bash
  python -m venv <virtual env>
  source <virtual env>/bin/activate
  pip install -r requirements.txt
```

Setup the server

- ```bash
  cd chatbot/
  python manage.py makemigrations
  python manage.py migrate
  python manage.py createsuperuser
  ```

- In the .env file write your OPENAI_KEY.

- On your browser go to the [link](http://127.0.0.1:8000/admin/) and create an instance of Patient.

- Now type you medical query in the [link](http://127.0.0.1:8000/),and wait for the response.


