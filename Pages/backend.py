from langchain_core.messages import HumanMessage
from typing import List
from dataclasses import dataclass
from langgraph.graph import StateGraph
from Pages.graph.state import AgentState
from Pages.graph.nodes import call_model, call_tools, route_to_tools
from Pages.data_models import InputData
from rich.console import Console
import sys
import os
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Literal
import operator
from pydantic import Field

console = Console()


@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"


class PythonChatbot:
    def __init__(self):
        super().__init__()
        self.reset_chat()
        self.graph = self.create_graph()

    def create_graph(self):
        console.log("Creating graph")
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.set_entry_point("agent")
        return workflow.compile()

    def user_sent_message(self, user_query, input_data: List[InputData]):
        console.log(f"User query: {user_query}\n\n")
        console.log(f"Input data: {input_data}\n\n")
        starting_image_paths_set = set(sum(self.output_image_paths.values(), []))
        input_state = {
            "messages": self.chat_history + [HumanMessage(content=user_query)],
            "output_image_paths": list(starting_image_paths_set),
            "input_data": input_data,
        }

        result = self.graph.invoke(input_state, {"recursion_limit": 25})
        self.chat_history = result["messages"]
        new_image_paths = set(result["output_image_paths"]) - starting_image_paths_set
        self.output_image_paths[len(self.chat_history) - 1] = list(new_image_paths)
        if "intermediate_outputs" in result:
            self.intermediate_outputs.extend(result["intermediate_outputs"])
            console.log(f"Intermediate outputs: {result['intermediate_outputs']}\n\n")

    def reset_chat(self):
        console.log("Resetting chat")
        self.chat_history = []
        self.intermediate_outputs = []
        self.output_image_paths = {}
