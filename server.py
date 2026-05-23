import asyncio
import json

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from entertainment_agent import create_entertainment_agent, lifespan
from onboarding_agent import create_onboarding_agent
from logging_utils import log_agent_steps
from models import CinemaRequest, AgentResponse, OnboardingRequest
from datetime import datetime

app = FastAPI(title="Entertainment Agent Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recommend", response_model=AgentResponse)
async def recommend(request: CinemaRequest):
    print(request.message)
    entertainment_agent = create_entertainment_agent()
    result = await entertainment_agent.run(
        request.message,
        expected_output=AgentResponse,
    )
    log_agent_steps(result.state)

    if result.output_structured is not None:
        return result.output_structured
    return AgentResponse.model_validate_json(result.last_message.text)

@app.get("/initial-message", response_model=OnboardingRequest)
async def get_initial_message():
    print("Getting initial message")
    onboarding_agent = create_onboarding_agent()
    result = await onboarding_agent.run(
        "Enviar un mensaje de bienvenida al usuario. La hora actual es: " + datetime.now().strftime("%H:%M"),
        expected_output=OnboardingRequest,
    )
    log_agent_steps(result.state)

    if result.output_structured is not None:
        return result.output_structured
    return OnboardingRequest.model_validate_json(result.last_message.text)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3004)
