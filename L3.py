import warnings

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool


warnings.filterwarnings('ignore')

## ------------------------------------------------------##
llm = LLM(model = "openai/gpt-4", max_tokens = 1024)

## ------------------------------------------------------##
config = {
        "llm" : {
                "provider" : "openai",
                "config" : {
                            "model" : "gpt-4",
                            }
                },
        "embedding_model" : {
                            "provider" : "openai",
                            "config" : {
                                        "model" : "text-embedding-ada-002"
                                        }
                            }
        }

## ------------------------------------------------------##
rag_tool = RagTool(config = config, chunk_size = 1200, chunk_overlap = 200, )

rag_tool.add("../data/gold-hospital-and-premium-extras.pdf", data_type = "pdf_file")

## ------------------------------------------------------##
rag_tool    # print()

## ------------------------------------------------------##
insurance_agent = Agent(
                        role = "Senior Insurance Coverage Assistant",
                        goal = "Determine whether something is covered or not",
                        backstory = "You are an expert insurance agent designed \
                                    to assist with coverage queries",
                        verbose = True,
                        allow_delegation = False,
                        llm = llm,
                        tools = [rag_tool],
                        max_retry_limit = 5
                        )

## ------------------------------------------------------##
task1 = Task(
            description = 'What is the waiting period for rehabilitation?',
            expected_output = "A comprehensive response as to the users question",
            agent = insurance_agent
            )

## ------------------------------------------------------##
crew = Crew(agents = [insurance_agent], tasks = [task1], verbose = True)
task_output = crew.kickoff()
print(task_output)
