FROM python

COPY . .

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "/app/Homepage.py", "--server.port 8080"]