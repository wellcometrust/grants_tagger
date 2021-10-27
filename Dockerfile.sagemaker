FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-gpu-py37-cu110-ubuntu18.04

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install sagemaker-training

ENV SAGEMAKER_PROGRAM grants_tagger/train.py
