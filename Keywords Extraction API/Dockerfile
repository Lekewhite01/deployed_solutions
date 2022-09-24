FROM python:3.9
LABEL maintainer "Olorunleke"
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
ENV PORT 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
