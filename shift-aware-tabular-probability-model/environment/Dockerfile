FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash agent \
    && passwd -l root

WORKDIR /app
COPY requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --no-deps -r /tmp/requirements.lock

COPY --chown=10001:10001 . /app
RUN chmod +x /app/build.sh /app/run.sh /app/test.sh \
    && chown -R 10001:10001 /app \
    && su -s /bin/sh agent -c 'cd /app && ./build.sh'

USER root
