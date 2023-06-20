FROM python:3.8.5

COPY . .

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "/app/Homepage.py"]