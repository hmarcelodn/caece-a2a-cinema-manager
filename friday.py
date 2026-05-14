import asyncio
import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement

profile_agent_location = "http://localhost:3007"
scout_agent_location = "http://localhost:3005"
research_agent_location = "http://localhost:3006"
weather_agent_location = "http://localhost:3002"

profile_agent = A2AAgent(url=profile_agent_location, memory=UnconstrainedMemory())
scout_agent = A2AAgent(url=scout_agent_location, memory=UnconstrainedMemory())
research_agent = A2AAgent(url=research_agent_location, memory=UnconstrainedMemory())
weather_agent = A2AAgent(url=weather_agent_location, memory=UnconstrainedMemory())


def create_friday_agent() -> RequirementAgent:
    return RequirementAgent(
        name="Friday Manager",
        description="Asistente personal de viernes a la noche que determina qué película ver, por qué, y dónde",
        llm=ChatModel.from_name("anthropic:claude-sonnet-4-20250514"),
        tools=[
            ThinkTool(),
            HandoffTool(
                target=profile_agent,
                name="Profile Agent",
                description=(
                    'Consultar las preferencias del usuario y su historial de películas vistas. '
                    'Acepta texto libre o JSON con esta forma: {"query":"¿qué géneros le gustan?"}. '
                    'Devuelve el perfil completo (géneros favoritos, plataformas, rating mínimo, historial) '
                    'junto con una respuesta puntual. Ejemplo: {"query":"dame el perfil completo"}'
                ),
            ),
            HandoffTool(
                target=weather_agent,
                name="Weather Agent",
                description=(
                    'Obtener las condiciones meteorológicas actuales en la ubicación del usuario. '
                    'El task DEBE ser un JSON estricto con esta forma: '
                    '{"location":{"latitude":N,"longitude":N}}. '
                    'Ejemplo: {"location":{"latitude":-34.61667,"longitude":-58.68333}}'
                ),
            ),
            HandoffTool(
                target=scout_agent,
                name="Scout Agent",
                description=(
                    'Buscar películas candidatas según preferencias estructuradas vía TMDB. '
                    'El task DEBE ser JSON estricto con esta forma: '
                    '{"preferences":{"genres":["thriller","comedy"],"minVoteAverage":7,"yearFrom":2000,"originalLanguage":"en","sortBy":"rating"},"limit":5}. '
                    'Los campos de preferences son todos opcionales excepto genres. '
                    'Usar la información obtenida del Profile Agent para armar las preferencias.'
                ),
            ),
            HandoffTool(
                target=research_agent,
                name="Movies Research Agent",
                description=(
                    'Investigar y recomendar películas basándose en el contexto completo del usuario. '
                    'Acepta texto libre en lenguaje natural describiendo: clima actual, estado de ánimo, '
                    'compañía, preferencias, películas ya vistas. Busca en internet (Tavily) y TMDB '
                    'para dar recomendaciones profundas con explicación de por qué cada película es ideal. '
                    'Ejemplo: "Está lloviendo en Buenos Aires, le gustan los thrillers y la ciencia ficción, '
                    'ya vio Inception e Interstellar, quiere algo para ver un viernes a la noche"'
                ),
            ),
        ],
        role="Asistente personal de entretenimiento",
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False),
        ],
        instructions=(
            f"""
    Sos el Friday Manager, un asistente personal que ayuda a elegir la película perfecta para un viernes a la noche.
    Tu objetivo es responder tres preguntas: QUÉ película ver, POR QUÉ esa película, y DÓNDE verla.

    Seguí este flujo de trabajo:

    1. PERFIL: Consultá al {profile_agent.name} para conocer las preferencias del usuario
       (géneros favoritos, plataformas de streaming, rating mínimo, películas ya vistas).

    2. CLIMA: Consultá al {weather_agent.name} con la ubicación del usuario (Buenos Aires: latitude -34.61667, longitude -58.68333)
       para saber las condiciones meteorológicas actuales. Esto te da contexto
       (ej: si llueve, recomendar algo para quedarse en casa; si hace lindo, quizás una película de aventura).

    3. BÚSQUEDA: Con las preferencias del perfil, armá un JSON estructurado y consultá al {scout_agent.name}
       para obtener candidatos de películas que cumplan los criterios del usuario.

    4. INVESTIGACIÓN: Consultá al {research_agent.name} pasándole en texto libre todo el contexto reunido:
       clima actual, preferencias, historial de películas vistas, y cualquier detalle relevante.
       Este agente busca en internet y TMDB para dar recomendaciones profundas y contextuales.

    5. SÍNTESIS: Con toda la información, elegí la mejor película y presentá tu recomendación final con:
       - QUÉ película ver (título, año, sinopsis breve)
       - POR QUÉ esa película (basándote en preferencias, clima, contexto del viernes)
       - DÓNDE verla (plataforma de streaming disponible, o cine si aplica)

    REGLAS:
    - SIEMPRE consultá al Profile Agent primero. No asumas preferencias.
    - SIEMPRE consultá al Weather Agent. El clima es contexto importante.
    - NO recomiendes películas que el usuario ya haya visto (chequeá el historial del perfil).
    - Respondé siempre en español.
    - Sé conciso pero informativo en tu recomendación final.
    """
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await profile_agent.check_agent_exists()
    print("Profile agent exists")
    await scout_agent.check_agent_exists()
    print("Scout agent exists")
    await research_agent.check_agent_exists()
    print("Movies Research agent exists")
    await weather_agent.check_agent_exists()
    print("Weather agent exists")
    yield


app = FastAPI(title="Friday Cinema Manager", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    message: str


def serialize_event(data, event_meta) -> str | None:
    event_name = event_meta.name
    path = event_meta.path

    if event_name == "start":
        return json.dumps({"step": "start", "path": path})
    elif event_name == "final_answer":
        delta = getattr(data, "delta", None)
        output = getattr(data, "output", None)
        return json.dumps({"step": "final_answer", "delta": delta or "", "output": output or ""})
    elif event_name == "success":
        return json.dumps({"step": "success", "path": path})
    else:
        tool_name = None
        if hasattr(data, "tool") and hasattr(data.tool, "name"):
            tool_name = data.tool.name
        elif hasattr(data, "state") and hasattr(data.state, "tool") and data.state.tool:
            tool_name = getattr(data.state.tool, "name", None)
        return json.dumps({"step": event_name, "path": path, "tool": tool_name})


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    friday_agent = create_friday_agent()

    async def event_stream():
        run = friday_agent.run(request.message)
        async for data, event_meta in run:
            payload = serialize_event(data, event_meta)
            if payload:
                yield f"event: {event_meta.name}\ndata: {payload}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3004)
