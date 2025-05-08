import os
import json
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState
from .tools import complete_python_task
from rich.console import Console


console = Console()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [complete_python_task]

model = llm.bind_tools(tools)


with open(
    os.path.join(os.path.dirname(__file__), "../prompts/main_prompt.md"), "r"
) as file:
    console.log(f"Loading prompt from {file.name}")
    prompt = file.read()

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("placeholder", "{messages}"),
    ]
)
model = chat_template | model


def create_data_summary(state: AgentState) -> str:
    console.print(f"[purple]Creating data summary: {state}[/]")
    summary = ""
    variables = []
    for d in state["input_data"]:
        variables.append(d.variable_name)
        summary += f"\n\nVariable: {d.variable_name}\n"
        summary += f"Description: {d.data_description}"

    if "current_variables" in state:
        remaining_variables = [
            v for v in state["current_variables"] if v not in variables
        ]
        for v in remaining_variables:
            summary += f"\n\nVariable: {v}"
    return summary


def route_to_tools(
    state: AgentState,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route back to the agent.
    """
    console.print(f"[blue]Routing to tools: {state}[/]")
    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


def call_model(state: AgentState):
    console.print(f"[cyan]Calling model: {state}[/]")

    current_data_template = """The following data is available:\n{data_summary}"""
    current_data_message = HumanMessage(
        content=current_data_template.format(data_summary=create_data_summary(state))
    )
    state["messages"] = [current_data_message] + state["messages"]

    llm_outputs = model.invoke(state)

    return {
        "messages": [llm_outputs],
        "intermediate_outputs": [current_data_message.content],
    }


def call_tools(state: AgentState):
    console.print(f"[green]Calling tools: {state}[/]")

    llm_outputs = model.invoke(state)

    # Check if the last message is a tool message
    if not isinstance(llm_outputs, ToolMessage):
        print("The last message is not a tool message.")

    return {
        "messages": [llm_outputs],
        "intermediate_outputs": [current_data_message.content],
    }

    # return llm_outputs
