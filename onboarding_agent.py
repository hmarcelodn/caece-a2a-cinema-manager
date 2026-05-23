from dotenv import load_dotenv

load_dotenv()

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.tools.think import ThinkTool

from logging_utils import build_middlewares
from models import OnboardingRequest


def create_onboarding_agent() -> RequirementAgent:
    return RequirementAgent(
        name="Onboarding Agent",
        description="Asistente personal para enviar un mensaje de bienvenida al usuario.",
        llm=ChatModel.from_name("anthropic:claude-haiku-4-5"),
        middlewares=build_middlewares(),
        tools=[
            # ThinkTool(),
        ],
        role="Asistente personal de onboarding",
        requirements=[],
        instructions=(
            f"""
            Sos Lumus, asistente para elegir películas.
            Tu objetivos son:
            - Presentate de manera amigable.
            - Enviar un mensaje de bienvenida al usuario teniendo en cuenta el momento del dia.
            - Saludar al usuario de manera amigable.            
            - El mensaje de bienvenida debe ser breve y directo.

            Retorna SOLO JSON válido con el schema de respuesta esperado:
            {OnboardingRequest.model_json_schema()}
            """
        ),
    )
