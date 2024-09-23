import streamlit as st
from streamlit_option_menu import option_menu

def sidebar():
    active_page = st.session_state.get('active_page', None)
    # Menu da sidebar usando option_menu
    with st.sidebar:
        selected_page = option_menu(
            "ControlGov",  # Título do menu
            ["Introdução","Artefatos","PMs & CMs/BA","CM Pinhão/SE", "Sobre"],  # Opções do menu
            icons=["house","clipboard","building","building", "info-circle"],  # Ícones para cada opção
            menu_icon="cast",  # Ícone do menu
            default_index=0,  # Opção selecionada por padrão
            styles={
                "nav-link-selected": {"background-color": "#1E3D59"},  # Cor da opção selecionada
                "nav-link": {"--hover-color": "#eee"}  # Cor ao passar o mouse
            },
        )
        return selected_page
