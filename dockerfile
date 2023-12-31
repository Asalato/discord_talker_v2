ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-alpine

WORKDIR /app

COPY . /app

RUN pip --no-cache-dir -r ./requirements.txt install
CMD ["python", "./main.py"]