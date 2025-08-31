from pydantic import BaseModel, Field

# Define the Request Schema
class SimilarityRequest(BaseModel):
    text1: str
    text2: str

# Define the Responce Schema
class SimilarityResponse(BaseModel):
    similarity_score: float = Field(..., alias="similarity score")

    class Config:
        populate_by_name = True 
