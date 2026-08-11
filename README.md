# SPS GenAI API

This repository contains the FastAPI code used across the class activities and assignments. For Assignment 4, it adds two CIFAR-10 image generation models to the Module 6 API:

- an Energy-Based Model (EBM)
- a small diffusion denoising model

The training code downloads CIFAR-10 with `torchvision.datasets.CIFAR10`, trains the model locally, and saves the model weights as `.pth` files. The API loads those weights at startup and exposes generation endpoints.

## Assignment 4 Checklist

- CIFAR-10 dataset: `train_energy_model.py` and `train_diffusion_model.py`
- Energy model: `app/energy_model.py`
- Diffusion model: `app/diffusion_model.py`
- API integration: `main.py`
- API endpoints: `POST /generate_energy_image` and `POST /generate_diffusion_image`

## Setup

Use Python 3.13 or newer.

```bash
uv sync
```

If you are not using `uv`, install the dependencies with pip:

```bash
python -m pip install fastapi[standard] spacy torch torchvision pillow
```

## Train the CIFAR-10 Models

Train the energy model:

```bash
python train_energy_model.py
```

This saves:

```text
energy_model.pth
```

Train the diffusion model:

```bash
python train_diffusion_model.py
```

This saves:

```text
diffusion_model.pth
```

The `.pth` files are ignored by Git because trained model files can be large. To run the API with trained models, create these files locally before starting the server.

## Run the API

```bash
fastapi dev main.py
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Assignment 4 Endpoints

Generate an image with the trained energy model:

```bash
curl -X POST "http://127.0.0.1:8000/generate_energy_image" \
  -H "Content-Type: application/json" \
  -d '{"steps": 30, "seed": 42}'
```

Generate an image with the trained diffusion model:

```bash
curl -X POST "http://127.0.0.1:8000/generate_diffusion_image" \
  -H "Content-Type: application/json" \
  -d '{"steps": 30, "seed": 42}'
```

If the trained weights are missing, these endpoints return a `503` response that explains which training script to run.

## Other API Features

The repository also includes earlier class activity endpoints for:

- bigram text generation: `POST /generate`
- RNN text generation: `POST /generate_with_rnn`
- RL post-trained text generation: `POST /generate_with_rl`
- word embeddings: `POST /embedding`
- word similarity: `POST /similarity`
- CIFAR-10 image classification: `POST /predict`
- MNIST-like GAN image generation: `GET /generate-digit`

## Assignment 5: Post-Training with RL

Assignment 5 uses reinforcement learning to post-train a text generation model so that its answer follows a required format. This implementation rewards outputs that:

- start with `That is a great question`
- end with `Let me know if you have any other questions.`
- contain enough generated content between the required opening and closing phrases

The RL code is in:

- `app/rl_text_model.py`
- `train_rl_text_model.py`

Train the RL post-trained text model:

```bash
python train_rl_text_model.py
```

This saves:

```text
rl_text_model.pth
```

The training script first warm-starts the small RNN text model with supervised next-token training, then applies a REINFORCE-style policy-gradient update using the formatting reward.

Run the API:

```bash
fastapi dev main.py
```

Generate a formatted answer:

```bash
curl -X POST "http://127.0.0.1:8000/generate_with_rl" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does reinforcement learning help text generation?", "max_words": 24, "seed": 42}'
```

Example response:

```json
{
  "question": "How does reinforcement learning help text generation?",
  "required_start": "That is a great question",
  "required_end": "Let me know if you have any other questions.",
  "generated_text": "That is a great question ... Let me know if you have any other questions.",
  "reward": 2.25
}
```

## Docker Deployment

The Docker image builds a FastAPI server and creates the Assignment 5 RL text model checkpoint during the image build.

Build the image:

```bash
docker build -t sps-genai .
```

Run the API:

```bash
docker run --rm -p 8000:80 sps-genai
```

Then query the Assignment 5 endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/generate_with_rl" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does reinforcement learning help text generation?", "max_words": 24, "seed": 42}'
```

## Repository Notes

- `data/` is ignored because CIFAR-10 is downloaded automatically by the training scripts.
- `*.pth` is ignored because trained weights are generated locally.
- Run the two Assignment 4 training scripts before testing the image generation endpoints.
- Run `train_rl_text_model.py` before testing the Assignment 5 RL text endpoint.
