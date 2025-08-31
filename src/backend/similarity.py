import os
import re
import string
import requests
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from src.utils.logger import logger

# Load env vars (Hugging Face token)
load_dotenv()

# Check for punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Check for stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Hugging Face API details
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"


class SemanticSimilarity:
    
    def __init__(self):
       
       try:

            # Load English stopwords
            self.stop_words = set(stopwords.words('english'))

            # Load SpaCy lemmatizer
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                raise Exception("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

            # Hugging Face headers
            if not HUGGINGFACEHUB_API_TOKEN:
                raise Exception("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in .env")
            self.headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

            logger.info("SemanticSimilarity initialized successfully")
        
       except Exception as e:
            logger.exception(f"Initialization failed: {e}")
            raise
           


    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, remove html/urls/punctuation,
        tokenize, remove stopwords, lemmatize.
        """
        logger.debug(f"Preprocessing text: {text[:50]}...")
        
        try:
            # 1. Lowercase
            text = text.lower()

            # 2. Remove HTML tags
            text = re.sub(r'<.*?>', ' ', text)

            # 3. Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)

            # 4. Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))

            # 5. Tokenize
            tokens = word_tokenize(text)

            # 6. Remove stopwords + non-alphabetic tokens
            tokens = [w for w in tokens if w not in self.stop_words and w.isalpha()]

            # 7. Lemmatize
            doc = self.nlp(" ".join(tokens))
            tokens = [token.lemma_ for token in doc]

            return " ".join(tokens)

        except Exception as e:
            logger.exception(f"Error in preprocessing: {e}")
            raise

    def _get_bge_embedding(self, sentences: list[str]) -> np.ndarray:
        """
        Get embeddings from Hugging Face Inference API using BAAI/bge-large-en-v1.5 model.
        """
        logger.debug(f"Fetching embeddings for {len(sentences)} sentence(s)")
        try:
            instructed_sentences = [
                f"Represent this sentence for searching relevant passages: {s}" for s in sentences
            ]
            response = requests.post(API_URL, headers=self.headers, json={"inputs": instructed_sentences})
            if response.status_code != 200:
                raise Exception(f"Hugging Face API Error: {response.text}")

            embeddings = np.array(response.json())
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            elif embeddings.ndim == 3:
                embeddings = np.array(embeddings[0])

            return embeddings

        except Exception as e:
            logger.exception(f"Embedding error: {str(e)}")
            raise

        
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Preprocess, embed, and compute cosine similarity between two texts.
        """
        logger.info("Calculating similarity score...")

        try:
            clean1 = self.preprocess_text(text1)
            clean2 = self.preprocess_text(text2)

            emb1 = self._get_bge_embedding([clean1])[0]
            emb2 = self._get_bge_embedding([clean2])[0]

            # Normalize
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)

            # Cosine similarity (same as normalized dot product)
            score = np.dot(emb1, emb2)

            score = round(float(score), 4)

            logger.debug(f"Similarity Score: {score}")
            return score

        except Exception as e:
            logger.exception(f"Error in get_similarity: {e}")
            raise

