import streamlit as st
import pandas as pd
import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from Pages.backend import PythonChatbot
from Pages.data_models import InputData
import pickle
from openai import OpenAI

# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

st.title("Data Analysis Dashboard")

# Load data dictionary
with open("data_dictionary.json", "r") as f:
    data_dictionary = json.load(f)

(tab1, tab2, tab3, tab4) = st.tabs(
    ["Data Management", "Original Chat", "Debug", "Development"]
)

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


with tab2:
    # Initialize session state

    def on_submit_user_query():
        user_query = st.session_state["user_input"]
        input_data_list = [
            InputData(
                variable_name=f"{file.split('.')[0]}",
                data_path=os.path.abspath(os.path.join("uploads", file)),
                data_description=data_dictionary.get(file, {}).get("description", ""),
            )
            for file in selected_files
        ]

        st.session_state.visualisation_chatbot.user_sent_message(
            user_query, input_data=input_data_list
        )

    if "selected_files" in st.session_state and st.session_state["selected_files"]:
        if "visualisation_chatbot" not in st.session_state:
            st.session_state.visualisation_chatbot = PythonChatbot()
        chat_container = st.container(height=None)
        with chat_container:
            # Display chat history with associated images
            for msg_index, msg in enumerate(
                st.session_state.visualisation_chatbot.chat_history
            ):
                msg_col, img_col = st.columns([2, 1])

                with msg_col:
                    if isinstance(msg, HumanMessage):
                        st.chat_message("You").markdown(msg.content)

                    elif isinstance(msg, AIMessage):
                        with st.chat_message("AI"):
                            st.markdown(msg.content)

                    if (
                        isinstance(msg, AIMessage)
                        and msg_index
                        in st.session_state.visualisation_chatbot.output_image_paths
                    ):
                        image_paths = (
                            st.session_state.visualisation_chatbot.output_image_paths[
                                msg_index
                            ]
                        )
                        for image_path in image_paths:
                            with open(
                                os.path.join(
                                    "images/plotly_figures/pickle", image_path
                                ),
                                "rb",
                            ) as f:
                                fig = pickle.load(f)
                            st.plotly_chart(fig, use_container_width=True)
        # Chat input
        st.chat_input(
            placeholder="Ask me anything about your data",
            on_submit=on_submit_user_query,
            key="user_input",
        )
    else:
        st.info("Please select files to analyze in the Data Management tab first.")
        # Update session state to scroll to bottom
    st.session_state.scroll_to_bottom = True
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
