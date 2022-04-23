FROM python:3.7-alpine

MAINTAINER Saeid Ghafouri "s.ghafouri@qmul.ac.uk"

# IMPORTANT NOTE: YOU HAVE TO PASS THE NAME/ADDRESS OF UTILIZATION_SERVER INTO ENVIRONMENT

# install required packages
RUN apk --no-cache --update add stress-ng

# install required libraries
RUN pip install flask requests

# copy files into container
COPY ./ /app

# change working directory to /app
WORKDIR /app

ENTRYPOINT python ./app.py