FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

RUN uv venv .venv
RUN uv pip install --python .venv/bin/python \
    "fastapi[standard]>=0.138.0" \
    "spacy>=3.8.14" \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    "pillow>=10.0.0"

COPY ./app /code/app
COPY main.py /code/
COPY train_rl_text_model.py /code/

RUN .venv/bin/python train_rl_text_model.py

EXPOSE 80

CMD [".venv/bin/fastapi", "run", "main.py", "--port", "80"]
