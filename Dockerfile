FROM python

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080/tcp

CMD ["streamlit", "run", "/app/Homepage.py", "--server.port 8080"]