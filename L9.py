import json
import requests
import os
import asyncio
import nest_asyncio

from IPython.display import IFrame
from colorama import Fore
from collections.abc import AsyncGenerator

from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server

from mcp import StdioServerParameters
from mcp.server.fastmcp import FastMCP

from smolagents import (CodeAgent, DuckDuckGoSearchTool, LiteLLMModel,
                        VisitWebpageTool, ToolCallingAgent, ToolCollection)


nest_asyncio.apply()

## ------------------------------------------------------##
mcp = FastMCP("doctorserver")


@mcp.tool()
def list_doctors(state: str) -> str:
    """This tool returns doctors that may be near you.
    Args:
        state: the two letter state code that you live in.
        Example payload: "CA"

    Returns:
        str: a list of doctors that may be near you
        Example Response "{"DOC001" : {"name" : "Dr John James", \
                                        "specialty" : "Cardiology"...}...}"
    """

    url = ('https://raw.githubusercontent.com/nicknochnack/ACPWalkthrough/refs/heads/main/'
            'doctors.json')
    resp = requests.get(url)
    doctors = json.loads(resp.text)

    matches = [doctor for doctor in doctors.values()
               if doctor['address']['state'] == state]

    return str(matches)


if __name__ == "__main__":
    mcp.run(transport = "stdio")

## ------------------------------------------------------##
server = Server()

model = LiteLLMModel(
                    model_id = "openai/gpt-4",
                    max_tokens = 2048
                    )

server_parameters = StdioServerParameters(
                                        command = "uv",
                                        args = ["run", "mcpserver.py"],
                                        env = None,
                                        )


@server.agent()
async def health_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """This is a CodeAgent which supports the hospital to handle health based questions
     for patients. Current or prospective patients can use it to find answers about their
     health and hospital treatments."""
    agent = CodeAgent(tools = [DuckDuckGoSearchTool(), VisitWebpageTool()], model = model)

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts = [MessagePart(content = str(response))])


@server.agent()
async def doctor_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a Doctor Agent which helps users find doctors near them."
    with ToolCollection.from_mcp(server_parameters, trust_remote_code = True) as tool_collection:
        agent = ToolCallingAgent(tools = [*tool_collection.tools], model = model)
        prompt = input[0].parts[0].content
        response = agent.run(prompt)

    yield Message(parts = [MessagePart(content = str(response))])


if __name__ == "__main__":
    server.run(port = 8000)

## ------------------------------------------------------##
url = os.environ.get('DLAI_LOCAL_URL').format(port = 8888)
IFrame(f"{url}terminals/2", width = 800, height = 600)

## ------------------------------------------------------##
async def run_doctor_workflow() -> None:
    async with Client(base_url = "http://localhost:8000") as hospital:
        run1 = await hospital.run_sync(
                                        agent = "doctor_agent",
                                        input = ("I'm based in Atlanta,GA. "
                                                "Are there any Cardiologists near me?")
                                        )

        content = run1.output[0].parts[0].content
        print(Fore.LIGHTMAGENTA_EX + content + Fore.RESET)

## ------------------------------------------------------##
asyncio.run(run_doctor_workflow())
