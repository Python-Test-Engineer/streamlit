import sys
import os
from langchain_core.tools import tool
# from langchain_experimental.utilities.python import PythonREPL
from langchain_core.messages import AIMessage
from typing import Annotated, Tuple
from langgraph.prebuilt import InjectedState
# import plotly.express as px
import pandas as pd
from rich.console import Console

console = Console()

persistent_vars = {}
plotly_saving_code = """import pickle
import uuid
import plotly

for figure in plotly_figures:
    pickle_filename = f"images/plotly_figures/pickle/{uuid.uuid4()}.pickle"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(figure, f)
"""


@tool(parse_docstring=True)
def complete_python_task(
    graph_state: Annotated[dict, InjectedState], thought: str, python_code: str
) -> Tuple[str, dict]:
    """Completes a python task

    Args:
        thought: Internal thought about the next action to be taken, and the reasoning behind it. This should be formatted in MARKDOWN and be high quality.
        python_code: Python code to be executed to perform analyses, create a new dataset or create a visualization.
    """
    console.print(f"[green]Thought: {thought}[/]")
    print(f"Python code: {python_code}")
    return "NONE", {}
