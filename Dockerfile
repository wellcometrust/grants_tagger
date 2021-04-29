FROM python:3.7

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install sagemaker-training

ENV SAGEMAKER_PROGRAM grants_tagger/train.py
