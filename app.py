import os

os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Set Streamlit to wide mode
st.set_page_config(layout="wide", page_title="Main Dashboard", page_icon="📊")

# Initialize session state
if "scroll_to_bottom" not in st.session_state:
    st.session_state.scroll_to_bottom = True

# Your chat interface code here...

# Update session state to scroll to bottom
st.session_state.scroll_to_bottom = True


data_visualisation_page = st.Page(
    "./Pages/python_visualisation_agent.py", title="Data Visualisation", icon="📈"
)

pg = st.navigation({"Visualisation Agent": [data_visualisation_page]})

pg.run()
