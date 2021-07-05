FROM python:3.7-slim
ENV PYTHONUNBUFFERED 1 

RUN apt-get update && apt-get install -y libssl-dev
RUN  apt-get install  -y gcc libpq-dev python3-dev python3-pip python3-venv python3-wheel curl
RUN apt-get -y upgrade

EXPOSE 8000 
WORKDIR /app 
# Copy requirements from host, to docker container in /app 
COPY ./requirements.txt .
# Copy everything from ./src directory to /app in the container
COPY . . 

RUN pip install -r requirements.txt 

# Run the application in the port 8000
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]