import os
import asyncio
import nest_asyncio

from IPython.display import IFrame
from acp_sdk.client import Client
from colorama import Fore


nest_asyncio.apply()

## ------------------------------------------------------##
url = os.environ.get('DLAI_LOCAL_URL').format(port =  8888)
IFrame(f"{url}terminals/1", width = 800, height= 600)

## ------------------------------------------------------##
async def example() -> None:
    async with Client(base_url  = "http://localhost:8001") as client:
        run = await client.run_sync(
                                    agent = "policy_agent",
                                    input = "What is the waiting period for rehabilitation?"
                                    )
        print(Fore.YELLOW + run.output[0].parts[0].content + Fore.RESET)

## ------------------------------------------------------##
asyncio.run(example())
