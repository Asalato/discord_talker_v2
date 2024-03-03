ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-alpine

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r ./requirements.txt
CMD ["python", "./main.py"]