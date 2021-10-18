FROM sbksinno/sbksinno:ubuntu20.04-python3.9

COPY requirements.app.txt requirements.app.txt

RUN pip install -r requirements.app.txt
