FROM python:3.12-slim

WORKDIR /app

COPY . /app
RUN python -m pip install --upgrade pip
RUN python -m pip --timeout 1000 install token_trace[client]

EXPOSE 8501
RUN python -m token_trace.load_pretrained_model
CMD ["streamlit", "run", "token_trace_app.py"]