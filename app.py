import os
import sys
import importlib

import streamlit as st
from streamlit_option_menu import option_menu
from app.services.text_functions import mkd_text_divider, mkd_text, mkd_paragraph

# Configuração da página com título e favicon
st.set_page_config(
    page_title="ControlGov",
    page_icon="./app/data/images/favicon_64_64.png",  # Caminho para o favicon
    layout="wide"
)

st.markdown(
    """
    <style>
    :root {
        --primaryColor: blue;
        --backgroundColor: white;
        --secondaryBackgroundColor: white;
        --textColor: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Adiciona o caminho da pasta de páginas ao sys.path
pages_dir = os.path.join(os.getcwd(), 'app', 'pages')
sys.path.append(pages_dir)

def show_banner():
    """Função para exibir a imagem do banner."""
    st.image("./app/data/images/Banner_1000_300_Azul_shaped.svg", use_container_width=True)

def load_css():
    """Função para injetar CSS para ocultar elementos e definir estilos."""
    css = """
        <style>
        /*#MainMenu {visibility: hidden;} /* Ocultar o menu principal do Streamlit */
        /*header {visibility: hidden;} /* Ocultar o cabeçalho inteiro */
        .st-emotion-cache-12fmjuu {padding-top: 0rem; margin-top: 0rem;} /* Reduz o padding e margin para o cabeçalho */
        .css-18e3th9 {padding-top: 0rem; padding-bottom: 0rem; background-color: white;} /* Ajusta o padding e define fundo da área principal */
        .css-1d391kg {background-color: white;} /* Define a cor de fundo da sidebar */
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def load_page(module_name):
    """Função para carregar dinamicamente um módulo de página."""
    try:
        imported_module = importlib.import_module(module_name)
        if hasattr(imported_module, 'run'):
            imported_module.run()
        else:
            st.write(f"O módulo {module_name} não possui uma função 'run()'.")
    except ModuleNotFoundError:
        st.error(f"Módulo {module_name} não encontrado em {pages_dir}.")
    except Exception as e:
        st.error(f"Erro ao carregar o módulo {module_name}: {e}")

def list_pages_directory():
    """Função para listar os arquivos no diretório de páginas para depuração."""
    try:
        files = os.listdir(pages_dir)
        st.write("Arquivos disponíveis em 'pages':", files)
    except Exception as e:
        st.error(f"Erro ao listar diretório 'pages': {e}")

def run():
    """Função para exibir a página 'Introdução'."""
    # Dividindo a página em duas colunas
    col1, col2 = st.columns([1,2])
    
    # Coluna 1: Logo da empresa
    with col1:
        # Usa o caminho absoluto para o arquivo
        image_path = os.path.abspath("./app/data/images/Logo_500_500.svg")
        st.image(image_path, width=230)
    
    # Coluna 2: Título e descrição centralizados
    with col2:
        st.markdown("<h1 style='text-align: center;'>ControlGov</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>FERRAMENTA DE APOIO AO CONTROLE INTERNO DO EIXO BAHIA|SERGIPE</h2>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("""
    <h6 style='text-align: justify;'>
        O Projeto Social ControlGov foi desenvolvido para auxiliar na resolução de desafios enfrentados pelos controles internos quanto à análise e monitoramento de dados financeiros governamentais. Apesar da disponibilidade desses dados em plataformas como Portais de Transparência e sistemas contábeis, a complexidade dessas ferramentas pode tornar a análise demorada, sujeita a erros e, às vezes, impraticável. O aplicativo busca oferecer uma interface intuitiva e utiliza algoritmos de análise e tecnologia de aprendizado de máquina para simplificar a visualização de despesas, categorizar credores e detectar anomalias, melhorando a transparência e eficiência no controle financeiro e orçamentário. Atuando há 8 anos no Controle Interno de uma casa legislativa, percebo e ouço as dificuldades enfrentadas por profissionais dessa área, tanto de prefeituras de grandes cidades quanto de câmaras municipais de cidades menores. Fiz deste projeto, que faz parte do meu trabalho de conclusão de curso para me formar como engenheiro de dados, minha contribuição para facilitar o trabalho desses profissionais e, de alguma forma, retribuir a ajuda que recebi durante todo esse período.
    </h6>
    """, unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>E-mail para contato: renoaldo@hotmail.com</h5>", unsafe_allow_html=True)
    st.divider()

def main():
    """Função principal para configurar o app."""
    load_css()
    from app.model.sidebar import sidebar
    selected_page = sidebar()

    # Função de depuração (remova após confirmar)
    #list_pages_directory()

    # Carregamento dinâmico de conteúdo com base na opção selecionada
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
    elif selected_page == "PMs & CMs/BA":
        show_banner()
        load_page("PMs_e_CMs")

if __name__ == "__main__":
    main()
