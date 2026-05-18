import asyncio
import json

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from agent import create_cinema_agent, lifespan
from models import CinemaRequest, AgentResponse

app = FastAPI(title="Cinema Agent Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recommend", response_model=AgentResponse)
async def recommend(request: CinemaRequest):
    cinema_agent = create_cinema_agent()
    result = await cinema_agent.run(
        request.message, 
        expected_output=AgentResponse
    )
    
    if result.output_structured is not None:
        return result.output_structured
    return AgentResponse.model_validate_json(result.last_message.text)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3004)
