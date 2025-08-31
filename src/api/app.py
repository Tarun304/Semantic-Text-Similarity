from src.api.routes import router
from fastapi import FastAPI
from src.utils.logger import logger

# Create the FastAPI app instance
app = FastAPI(
    title="Semantic Text Similarity API",
    description="Compute semantic similarity score between two texts.",
    docs_url="/docs",
    
)

logger.info("FastAPI app initialized")

# Include the routes
app.include_router(router, prefix='/api')

# Health Check
@app.get("/")
def root():
    """Health check endpoint for the API."""
    return {"message": "Text Similarity API is running 🚀"}