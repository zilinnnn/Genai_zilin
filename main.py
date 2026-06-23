from fastapi import FastAPI
from pydantic import BaseModel

from app.bigram_model import BigramModel
import spacy

nlp = spacy.load("en_core_web_lg")

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

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