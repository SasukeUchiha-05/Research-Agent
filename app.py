from fastapi import FastAPI
from pydantic import BaseModel
from research_agent import run_research_agent
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request model
class ResearchRequest(BaseModel):
    query: str

@app.post("/research")
async def research(request: ResearchRequest):
    """
    Endpoint to handle research requests.
    """
    response = run_research_agent(request.query)
    return response.dict()  # Convert Pydantic object to JSON
