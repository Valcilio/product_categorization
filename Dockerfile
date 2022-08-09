FROM python:3.10.4

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000

COPY ./main.py /app/main.py

COPY ./tests /app/tests

COPY ./domain /app/domain

COPY ./models /app/models

COPY ./scalers /app/scalers

RUN pytest

ENTRYPOINT ["python"]

CMD ["main.py"]
