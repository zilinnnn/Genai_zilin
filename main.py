from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.rnn_model import RNNTextGenerator
from app.energy_model import EnergyBasedImageGenerator
from app.diffusion_model import DiffusionImageGenerator
import spacy
import torch
from PIL import Image
import torchvision.transforms as transforms
from app.cnn_model import AssignmentCNN
from app.gan_model import Generator

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.blank("en")

app = FastAPI()

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",   
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

cnn_model = AssignmentCNN().to(device)
try:
    cnn_model.load_state_dict(
        torch.load("cnn_cifar10.pth", map_location=device)
    )
    cnn_model.eval()
except (FileNotFoundError, RuntimeError) as error:
    cnn_model = None
    print(f"CNN model not loaded: {error}")

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
rnn_model = RNNTextGenerator(corpus, device=torch.device("cpu"))
rnn_model.train_model()
energy_generator = EnergyBasedImageGenerator(device=torch.device("cpu"))
diffusion_generator = DiffusionImageGenerator(device=torch.device("cpu"))


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class ImageGenerationRequest(BaseModel):
    steps: int = 30
    seed: int | None = None

class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {
        "generated_text": generated_text
    }


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated_text = rnn_model.generate_text(
        request.start_word,
        request.length
    )

    return {
        "generated_text": generated_text
    }

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):

    embedding = nlp(request.word).vector

    return {
        "word": request.word,
        "embedding": embedding[:20].tolist()
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):

    similarity = nlp(request.word1).similarity(
        nlp(request.word2)
    )

    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": float(similarity)
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if cnn_model is None:
        raise HTTPException(
            status_code=503,
            detail="CNN model weights are not available."
        )

    image = Image.open(file.file).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    class_index = predicted.item()
    class_name = classes[class_index]

    return {
        "class_index": class_index,
        "class_name": class_name
    }


@app.post("/generate_energy_image")
def generate_energy_image(request: ImageGenerationRequest):
    result = energy_generator.generate(
        steps=request.steps,
        seed=request.seed
    )

    return {
        "message": "Generated a CIFAR-sized image using EBM sampling",
        **result
    }


@app.post("/generate_diffusion_image")
def generate_diffusion_image(request: ImageGenerationRequest):
    result = diffusion_generator.generate(
        steps=request.steps,
        seed=request.seed
    )

    return {
        "message": "Generated a CIFAR-sized image using diffusion sampling",
        **result
    }



   

gan_generator = Generator(z_dim=100).to(device)

try:
    gan_generator.load_state_dict(
        torch.load("gan_generator.pth", map_location=device)
    )
    gan_generator.eval()
    print("GAN generator loaded.")
except (FileNotFoundError, RuntimeError) as error:
    print(f"GAN model not loaded: {error}")
    print("Using untrained generator.")


@app.get("/generate-digit")
def generate_digit():
    gan_generator.eval()

    noise = torch.randn(1, 100).to(device)

    with torch.no_grad():
        fake_image = gan_generator(noise).cpu()

    image_array = fake_image.squeeze().tolist()

    return {
        "message": "Generated one MNIST-like digit image",
        "image_shape": [1, 28, 28],
        "image": image_array
    }
