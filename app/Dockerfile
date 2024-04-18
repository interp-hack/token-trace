FROM python:3.12-slim

WORKDIR /app

COPY . /app
RUN python -m pip install --upgrade pip
RUN python -m pip --timeout 1000 install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "token_trace_app.py"]