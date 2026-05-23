import sys

from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from logging_utils import build_middlewares
from models import AgentResponse

profile_agent_location = "http://localhost:3007"
scout_agent_location = "http://localhost:3005"
research_agent_location = "http://localhost:3006"
weather_agent_location = "http://localhost:3002"

profile_agent = A2AAgent(url=profile_agent_location, memory=UnconstrainedMemory())
scout_agent = A2AAgent(url=scout_agent_location, memory=UnconstrainedMemory())
research_agent = A2AAgent(url=research_agent_location, memory=UnconstrainedMemory())
weather_agent = A2AAgent(url=weather_agent_location, memory=UnconstrainedMemory())

def create_entertainment_agent() -> RequirementAgent:
    return RequirementAgent(
        name="Cinema Agent",
        description="Asistente personal de viernes a la noche que determina qué película ver, por qué, y dónde",
        llm=ChatModel.from_name("anthropic:claude-sonnet-4-20250514"),
        middlewares=build_middlewares(),
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
    Sos Lumus, un asistente personal que ayuda a elegir la película perfecta o ayuda a resolver cualquier consulta sobre peliculas.
    
    Tus objetivos son:
    - Responder tres preguntas: QUÉ película ver, POR QUÉ esa película, y DÓNDE verla.
    - Responder en que plataformas está disponible una película específica.
    - Responder cualquier información sobre una película específica.
    
    Cuando recomiendes una película obtener todo el contexto necesario para responder la pregunta.
    El contexto necesario para recomendar una película es:    
    - Clima/tiempo: "está lloviendo", "hace mucho calor", "día nublado"
    - Día/momento: "es viernes a la noche", "domingo de tarde", "feriado"
    - Estado de ánimo: "estoy aburrido", "quiero algo liviano", "tengo ganas de llorar"
    - Preferencias: géneros favoritos, actores, directores, épocas
    - Historial: películas que ya vio (para evitar repetir y encontrar patrones)
    - Compañía: "voy a ver con mi pareja", "noche con amigos", "para ver con los chicos"
    - Plataformas de streaming: "netflix", "amazon prime", "disney+", etc.
    - Idiomas: "español", "inglés", etc.
    - Horario: "de 10:00 a 12:00", "de 12:00 a 14:00", etc.
    
    Las siguientes herramientas estan disponibles para obtener el contexto necesario:
    - Consulta {weather_agent.name} para obtener el clima.
    - Consulta {scout_agent.name} para buscar películas candidatas.
    - Consulta {research_agent.name} para investigar y recomendar películas.
    - Consulta {profile_agent.name} para obtener el perfil del usuario.
    
    Retorna SOLO JSON válido con el schema de respuesta esperado:
    {AgentResponse.model_json_schema()}
    """
        )
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
