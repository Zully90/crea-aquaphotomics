FROM python

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:."

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "/app/Homepage.py"]