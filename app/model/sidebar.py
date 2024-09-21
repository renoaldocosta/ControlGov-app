import streamlit as st
from streamlit_option_menu import option_menu

def sidebar():
    active_page = st.session_state.get('active_page', None)
    # Sidebar menu using option_menu
    with st.sidebar:
        selected_page = option_menu(
            "ControlGov",  # Menu title
            ["Introdução","Artefatos","PM Pojuca/BA","CM Pinhão/SE", "Sobre","teste"  ],  # Menu options including Home
            icons=["house","clipboard","building","building", "info-circle", "info-circle"],  # Icons for each option
            menu_icon="cast",  # Icon for the menu
            default_index=0,  # Default selected option
            styles={
                "nav-link-selected": {"background-color": "#1E3D59"},  # Change selected option color
                "nav-link": {"--hover-color": "#eee"}  # Optional: change hover color
            },
        )
        return selected_page
    
