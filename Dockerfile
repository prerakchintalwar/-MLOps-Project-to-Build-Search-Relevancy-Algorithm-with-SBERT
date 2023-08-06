###############
# BUILD IMAGE #
###############
FROM python:3.8 

# disables lag in stdout/stderr output
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# add and install requirements
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# set working directory
RUN mkdir /app
WORKDIR /app

EXPOSE 5001
