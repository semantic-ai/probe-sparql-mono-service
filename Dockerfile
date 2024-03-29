FROM python:3.11

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data all
ENV NLTK_DATA=/usr/local/share/nltk_data

COPY src src
