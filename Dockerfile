FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    PORT=8501

WORKDIR /app
COPY requirements-ui.txt ./requirements-ui.txt
RUN pip install --no-cache-dir -r requirements-ui.txt \
    && python -m spacy download en_core_web_sm \
    && python -m nltk.downloader -d "$NLTK_DATA" \
       stopwords wordnet omw-1.4 punkt punkt_tab \
       averaged_perceptron_tagger_eng vader_lexicon

COPY wellness ./wellness
COPY app.py ./app.py
COPY data ./data
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
