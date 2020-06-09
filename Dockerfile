FROM python:3.7-slim-buster

RUN pip install --upgrade pip

# Install requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy Source
COPY src /src
COPY models /models

RUN python src/train.py

CMD ["python", "src/run.py"]