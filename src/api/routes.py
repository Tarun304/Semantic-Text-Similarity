from fastapi import APIRouter
from src.api.models import SimilarityRequest, SimilarityResponse
from src.backend.similarity import SemanticSimilarity
from src.utils.logger import logger

# Create the router object
router=APIRouter()

# Initialize SemanticSimilarity model
sim_model=SemanticSimilarity()


# Define the endpoint
@router.post('/similarity', response_model=SimilarityResponse)
def calculate_similarity(request: SimilarityRequest):
    """
    Compute semantic similarity between two input texts.
    Returns a JSON response with {"similarity score": float}.
    """
    try:
        logger.info(f"Received request: text1='{request.text1[:30]}...', text2='{request.text2[:30]}...'")
        similarity_score = sim_model.get_similarity(request.text1, request.text2)
        logger.info(f"Returning similarity score: {similarity_score}")
        return SimilarityResponse(similarity_score=similarity_score)
    except Exception as e:
        logger.exception(f"Error in /similarity endpoint: {e}")
        raise