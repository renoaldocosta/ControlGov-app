import streamlit as st
import os
import sys
from streamlit_option_menu import option_menu

from app.services.text_functions import mkd_text_divider, mkd_text, mkd_paragraph


# Set the page configuration with title and favicon
st.set_page_config(
    page_title="ControlGov",
    page_icon="./app/data/images/favicon_64_64.png",  # Path to your favicon file
    layout="wide"
)

# Add the path of your pages folder to sys.path
pages_dir = os.path.join(os.getcwd(), 'app', 'pages')
sys.path.append(pages_dir)

def show_banner():
    """Function to display the banner image."""
    st.image("./app/data/images/Banner_1000_300_Azul_shaped.svg", use_column_width=True)

def load_css():
    """Function to inject CSS for hiding elements and setting styles."""
    css = """
        <style>
        #MainMenu {visibility: hidden;} /* Hide the Streamlit main menu */
        header {visibility: hidden;} /* Hide the entire header */
        .st-emotion-cache-12fmjuu {padding-top: 0rem; margin-top: 0rem;} /* Reduce padding and margin for the header */
        .css-18e3th9 {padding-top: 0rem; padding-bottom: 0rem; background-color: white;} /* Adjust the padding and set background of the main content area */
        .css-1d391kg {background-color: white;} /* Set the background color of the sidebar */
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def load_page(module_name):
    """Function to dynamically load a page module."""
    try:
        imported_module = __import__(module_name)
        if hasattr(imported_module, 'run'):
            imported_module.run()
        else:
            st.write(f"The module {module_name} does not have a 'run()' function.")
    except ModuleNotFoundError:
        st.error(f"Module {module_name} not found in {pages_dir}.")



def run():

    # Dividindo a página em duas colunas
    col1, col2 = st.columns([1,2])
    
    # Coluna 1: Logo da empresa
    with col1:
        # Usa o caminho absoluto para o arquivo
        image_path = os.path.abspath("./app/data/images/Logo_500_500.svg")
        st.image(image_path, width=230)
        #st.image("../ControlGov/app/data/images/Logo_500_500.png", width=230)
    
    # Coluna 2: Título e descrição centralizados
    with col2:
        st.markdown("<h1 style='text-align: center;'>ControlGov</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>FERRAMENTA DE APOIO AO CONTROLE INTERNO DO EIXO BAHIA|SERGIPE</h2>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("<h6 style='text-align: justify;'>O Projeto Social ControlGov foi desenvolvido para auxiliar na resolução de desafios enfrentados pelos controles internos quanto à análise e monitoramento de dados financeiros governamentais. Apesar da disponibilidade desses dados em plataformas como Portais de Transparência e sistemas contábeis, a complexidade dessas ferramentas pode tornar a análise demorada, sujeita a erros e, às vezes, impraticável. O aplicativo busca oferecer uma interface intuitiva e utiliza algoritmos de análise e tecnologia de aprendizado de máquina para simplificar a visualização de despesas, categorizar credores e detectar anomalias, melhorando a transparência e eficiência no controle financeiro e orçamentário. Atuando há 8 anos no Controle Interno de uma casa legislativa, percebo e ouço as dificuldades enfrentadas por profissionais dessa área, tanto de prefeituras de grandes cidades quanto de câmaras municipais de cidades menores. Fiz deste projeto, que faz parte do meu trabalho de conclusão de curso para me formar como engenheiro de dados, minha contribuição para facilitar o trabalho desses profissionais e, de alguma forma, retribuir a ajuda que recebi durante todo esse período.</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>E-mail para contato: renoaldo@hotmail.com</h5>", unsafe_allow_html=True)
    st.divider()
    
    
def main():
    """Main function to set up the app."""
    load_css()
    from app.model.sidebar import sidebar
    selected_page = sidebar()

    # Dynamic content loading based on the selected menu option
    if selected_page == "Introdução":
        run()
    elif selected_page == "Sobre":
        show_banner()
        load_page("Sobre")
    elif selected_page == "Artefatos":
        show_banner()
        load_page("Artefatos")
    elif selected_page == "CM Pinhão/SE":
        show_banner()
        load_page("CM_Pinhao_SE")
    elif selected_page == "PM Pojuca/BA":
        show_banner()
        load_page("PM_Pojuca_BA")
    elif selected_page == "teste":
        show_banner()
        load_page("teste_mongo")

if __name__ == "__main__":
    main()

