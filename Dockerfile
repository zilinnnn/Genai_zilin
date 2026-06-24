FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock /code/

RUN uv sync --frozen

COPY ./app /code/app
COPY main.py /code/

EXPOSE 80

CMD ["uv", "run", "fastapi", "run", "main.py", "--port", "80"]