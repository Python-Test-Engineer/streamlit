import os
import json
import operator
from typing import Dict
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
from langgraph.graph import StateGraph
import pickle
from openai import OpenAI
from dataclasses import dataclass
from rich.console import Console
from typing import Sequence, TypedDict, Annotated, List, Literal, Union
from langchain_core.messages import BaseMessage


@dataclass
class InputData:
    variable_name: str
    data_path: str
    data_description: str


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    input_data: Annotated[List[InputData], operator.add]
    intermediate_outputs: Annotated[List[dict], operator.add]
    current_variables: dict
    output_image_paths: Annotated[List[str], operator.add]


console = Console()


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


# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

st.title("Data Analysis Dashboard")

# Load data dictionary
with open("data_dictionary.json", "r") as f:
    data_dictionary = json.load(f)

(tab1, tab3, tab4) = st.tabs(["Data Management", "Debug", "Development"])

# Tab 1: Data Management
with tab1:
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload CSV files", type="csv", accept_multiple_files=True
    )

    if uploaded_files:
        # Save uploaded files
        for file in uploaded_files:
            with open(os.path.join("uploads", file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success("Files uploaded successfully!")

    # Get list of available CSV files
    available_files = [f for f in os.listdir("uploads") if f.endswith(".csv")]

    if available_files:
        # File selection
        selected_files = st.multiselect(
            "Select files to analyze", available_files, key="selected_files"
        )

        # Dictionary to store new descriptions
        new_descriptions = {}

        if selected_files:
            # Create tabs for each selected file
            file_tabs = st.tabs(selected_files)

            # Display dataframe previews and data dictionary info in tabs
            for tab, filename in zip(file_tabs, selected_files):
                with tab:
                    try:
                        df = pd.read_csv(os.path.join("uploads", filename))
                        st.write(f"Preview of {filename}:")
                        st.dataframe(df.head())

                        # Display/edit data dictionary information
                        st.subheader("Dataset Information")

                        if filename in data_dictionary:
                            info = data_dictionary[filename]
                            current_description = info.get("description", "")
                        else:
                            current_description = ""

                        new_descriptions[filename] = st.text_area(
                            "Dataset Description",
                            value=current_description,
                            key=f"description_{filename}",
                            help="Provide a description of this dataset",
                        )

                        if filename in data_dictionary:
                            info = data_dictionary[filename]

                            if "coverage" in info:
                                st.write(f"**Coverage:** {info['coverage']}")

                            if "features" in info:
                                st.write("**Features:**")
                                for feature in info["features"]:
                                    st.write(f"- {feature}")

                            if "usage" in info:
                                st.write("**Usage:**")
                                if isinstance(info["usage"], list):
                                    for use in info["usage"]:
                                        st.write(f"- {use}")
                                else:
                                    st.write(f"- {info['usage']}")

                            if "linkage" in info:
                                st.write(f"**Linkage:** {info['linkage']}")

                    except Exception as e:
                        st.error(f"Error loading {filename}: {str(e)}")

            # Save button for descriptions
            if st.button("Save Descriptions"):
                for filename, description in new_descriptions.items():
                    if description:  # Only update if description is not empty
                        if filename not in data_dictionary:
                            data_dictionary[filename] = {}
                        data_dictionary[filename]["description"] = description

                # Save updated data dictionary
                with open("data_dictionary.json", "w") as f:
                    json.dump(data_dictionary, f, indent=4)
                st.success("Descriptions saved successfully!")

    else:
        st.info("No CSV files available. Please upload some files first.")


with tab3:
    if "visualisation_chatbot" in st.session_state:
        st.subheader("Intermediate Outputs")
        for i, output in enumerate(
            st.session_state.visualisation_chatbot.intermediate_outputs
        ):
            with st.expander(f"Step {i+1}"):
                if "thought" in output:
                    st.markdown("### Thought Process")
                    st.markdown(output["thought"])
                if "code" in output:
                    st.markdown("### Code")
                    st.code(output["code"], language="python")
                if "output" in output:
                    st.markdown("### Output")
                    st.text(output["output"])
                else:
                    st.markdown("### Output")
                    st.text(output)
        st.info(
            "No debug information available yet. Start a conversation to see intermediate outputs."
        )

with tab4:
    TITLE = "Pandas Data Analyst AI Copilot"

    # ---------------------------
    # Streamlit App Configuration
    # ---------------------------

    st.title(TITLE)

    st.markdown(
        """
    Welcome to the Pandas Data Analyst AI. Upload a CSV or Excel file and ask questions about the data.  
    The AI agent will analyze your dataset and return either data tables or interactive charts.
    """
    )

    with st.expander("Example Questions", expanded=False):
        st.write(
            """
            ##### Bikes Data Set:
            
            -  Show the top 5 bike models by extended sales.
            -  Show the top 5 bike models by extended sales in a bar chart.
            -  Show the top 5 bike models by extended sales in a pie chart.
            -  Make a plot of extended sales by month for each bike model. Use a color to identify the bike models.
            """
        )

    # ---------------------------
    # OpenAI API Key Entry and Test
    # ---------------------------

    # st.sidebar.header("Enter your OpenAI API Key")

    st.session_state["OPENAI_API_KEY"] = os.getenv(
        "OPENAI_API_KEY", st.session_state.get("OPENAI_API_KEY", "")
    )

    # Test OpenAI API Key
    if st.session_state["OPENAI_API_KEY"]:
        # Set the API key for OpenAI
        client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

        # Test the API key (optional)
        try:
            # Example: Fetch models to validate the key
            models = client.models.list()
            st.success("API Key is valid!")
        except Exception as e:
            st.error(f"Invalid API Key: {e}")
    else:
        st.info("Please enter your OpenAI API Key to proceed.")
        st.stop()
