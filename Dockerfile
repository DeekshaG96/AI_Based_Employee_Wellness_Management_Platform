FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    PORT=8501

WORKDIR /app

COPY pyproject.toml README.md ./
COPY wellness ./wellness
COPY app.py ./

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[ui]" \
    && python -m nltk.downloader -d "${NLTK_DATA}" \
       stopwords wordnet punkt punkt_tab \
       averaged_perceptron_tagger averaged_perceptron_tagger_eng vader_lexicon

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
