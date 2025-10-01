import logging
import os

from IPython.display import IFrame
from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool
from dotenv import load_dotenv


load_dotenv()

## ------------------------------------------------------##
server = Server()

model = LiteLLMModel(
                    model_id = "openai/gpt-4",
                    max_tokens = 2048
                    )


@server.agent()
async def health_agent(input: list[Message],
                       context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    This is a CodeAgent which supports the hospital to handle health based questions \
    for patients. Current or prospective patients can use it to find answers about their \
    health and hospital treatments.
    """

    agent = CodeAgent(tools = [DuckDuckGoSearchTool(), VisitWebpageTool()], model = model)

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts = [MessagePart(content = str(response))])


if __name__ == "__main__":
    server.run(port = 8000)

## ------------------------------------------------------##
url = os.environ.get('DLAI_LOCAL_URL').format(port = 8888)
IFrame(f"{url}terminals/2", width = 800, height = 600)
