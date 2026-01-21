FROM python:3.11-slim-trixie AS python-base

RUN apt update && apt -y install pipx && apt clean
RUN pipx install poetry==2.3
ENV PATH="/root/.local/bin:${PATH}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app
COPY . /app
RUN poetry install --extras serve

CMD ["poetry", "run", "gunicorn", "PASCal.app:app", "--bind", "0.0.0.0:5000", "--workers", "1"]
