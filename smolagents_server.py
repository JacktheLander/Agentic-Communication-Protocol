from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool   # Smolagent will use gpt for just a single step rather than a whole iterative process like a crewai agent
import logging
from mcp import StdioServerParameters

server = Server()

model = LiteLLMModel(
    model_id="openai/gpt-4",  
    max_tokens=2048
)

@server.agent()
async def health_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a CodeAgent which supports the hospital to handle health based questions for patients. Current or prospective patients can use it to find answers about their health and hospital treatments."
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=model)  # CodeAgent is specifically for writing code or steps of a process

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])

@server.agent()
async def doctor_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a Doctor Agent which helps users find doctors near them."
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
        prompt = input[0].parts[0].content
        response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])

if __name__ == "__main__":
    server.run(port=8000)
