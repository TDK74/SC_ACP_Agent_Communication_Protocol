import os
import nest_asyncio
import asyncio

from acp_sdk.client import Client
from colorama import Fore
from IPython.display import IFrame


nest_asyncio.apply()

## ------------------------------------------------------##
url = os.environ.get('DLAI_LOCAL_URL').format(port = 8888)
IFrame(f"{url}terminals/1", width = 800, height = 600)

## ------------------------------------------------------##
url = os.environ.get('DLAI_LOCAL_URL').format(port = 8888)
IFrame(f"{url}terminals/2", width = 800, height = 600)

## ------------------------------------------------------##
async def run_hospital_workflow() -> None:
    async with (Client(base_url = "http://localhost:8001") as insurer,
                Client(base_url = "http://localhost:8000") as hospital):
        run1 = await hospital.run_sync(
                                    agent = "health_agent",
                                    input = ("Do I need rehabilitation after "
                                             "a shoulder reconstruction?")
                                    )

        content = run1.output[0].parts[0].content
        print(Fore.LIGHTMAGENTA_EX + content + Fore.RESET)

        run2 = await insurer.run_sync(
                                    agent = "policy_agent",
                                    input = (f"Context: {content} What is the waiting "
                                             f"period for rehabilitation?")
                                    )

        print(Fore.YELLOW + run2.output[0].parts[0].content + Fore.RESET)

## ------------------------------------------------------##
asyncio.run(run_hospital_workflow())

