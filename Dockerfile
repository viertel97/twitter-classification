ARG PYTHON_BASE=3.11-slim-buster

FROM python:$PYTHON_BASE AS builder
ENV BLIS_ARCH=generic

RUN pip install -U pdm

COPY pyproject.toml pdm.lock README.md /project/
COPY src/ /project/src

WORKDIR /project
RUN pdm install --check --prod --no-editable -v

FROM python:$PYTHON_BASE

COPY --from=builder /project/.venv/ /project/.venv
ENV PATH="/project/.venv/bin:$PATH"
ENV PYTHONPATH="/project"

COPY src /project/src

RUN python -m spacy download en_core_web_trf

EXPOSE 80

CMD ["python", "/project/src/main.py"]