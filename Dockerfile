FROM python:3.10.9

WORKDIR /usr/src/app

EXPOSE 8000

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

CMD cd ./web/website && gunicorn website.wsgi:application --bind 0.0.0.0:8000
