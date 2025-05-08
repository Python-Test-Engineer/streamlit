import streamlit as st


# --- PAGE SETUP ---
about_page = st.Page(
    "views/about_me.py",
    title="Meet Trevor",
    icon=":material/account_circle:",
    default=True,
)
dashboard = st.Page(
    "views/sales_dashboard.py",
    title="Sales Dashboard",
    icon=":material/bar_chart:",
)
chatbot = st.Page(
    "views/chatbot.py",
    title="Chat Bot",
    icon=":material/smart_toy:",
)
data = st.Page(
    "views/ai.py",
    title="Data Management",
    icon=":material/smart_toy:",
)

# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "About": [about_page],
        "Tools": [dashboard, chatbot, data],
    }
)


# --- SHARED ON ALL PAGES ---
st.logo("assets/robot.png")
st.sidebar.markdown("Made with ❤️ by Craig West")


# --- RUN NAVIGATION ---
pg.run()
