import sys
import os
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Literal
import operator
from pydantic import Field

# from langchain_experimental.utilities.python import PythonREPL
from langchain_core.messages import AIMessage
from typing import Annotated, Tuple
from langgraph.prebuilt import InjectedState

# import plotly.express as px
import pandas as pd
from rich.console import Console

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
