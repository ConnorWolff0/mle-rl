FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash agent

WORKDIR /app
COPY --chown=10001:10001 . /app

RUN chmod +x /app/build.sh /app/run.sh /app/test.sh \
    && python3 -m compileall -q /app/policy.py /app/serving_sim /app/tests \
    && chown -R 10001:10001 /app

USER root
