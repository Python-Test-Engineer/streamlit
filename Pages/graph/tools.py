from langchain_core.tools import tool

# from langchain_experimental.utilities.python import PythonREPL
from langchain_core.messages import AIMessage
from typing import Annotated, Tuple
from langgraph.prebuilt import InjectedState
import sys
from io import StringIO
import os

# import plotly.graph_objects as go
# import plotly.io as pio
# import plotly.express as px
import pandas as pd

# import sklearn


# repl = PythonREPL()

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

    return "NONE", {}
