import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys
import pandas as pd

# Set the page configuration with title and favicon
st.set_page_config(
    page_title="ControlGov",
    page_icon="./app/data/images/favicon_64_64.png",  # Path to your favicon file
    layout="wide"
)

# Add the path of your pages folder to sys.path
pages_dir = os.path.join(os.getcwd(), 'app', 'pages')
sys.path.append(pages_dir)

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

def show_dataset():
    """Function to display the dataset."""
    st.write("###  Amostra dos dados:")
    data = pd.read_json("./app/data/dataset/Despesa - Prefeitura Municipal de POJUCA.json")
    st.dataframe(data)

def show_banner():
    """Function to display the banner image."""
    st.image("./app/data/images/Banner_1000_300.svg", use_column_width=True)

def show_line():
    """Function to display the a transparent image."""
    st.image("./app/data/images/line_2000_40.png", use_column_width=True)
    
def home_page():
    """Content for the Home Page."""
    st.title("ControlGov")
    st.write("#### Problemas Identificados:")
    st.write("""
        Embora existam plataformas como os Portais de Transparência e Órgãos Centrais de Governo que disponibilizam dados financeiros, a análise e interpretação eficaz desses dados ainda representam um desafio significativo para gestores, cidadãos e para os próprios Controles Internos. Além disso, os sistemas de contabilidade atualmente utilizados nesses órgãos são muito complexos, o que torna o processo de análise e monitoramento muito mais demorado, sujeito a erros e, em alguns casos, impraticáveis. Este cenário pode resultar nos seguintes problemas:

        **1. Incapacidade de Realizar Atribuições Básicas:** Devido à quantidade e complexidade das operações financeiras, muitas vezes os Controles Internos sequer realizam as funções básicas de monitoramento e controle, ou realizam em uma escala bastante reduzida. A sobrecarga de trabalho, combinada com a falta de ferramentas adequadas, resulta em uma baixa efetividade das ações de controle e em dificuldades para cumprir os prazos necessários, comprometendo a eficiência e a qualidade do controle financeiro.

        **2. Análises Complexas e Demoradas:** Os Controles Internos enfrentam dificuldade em obter uma visão clara e imediata das despesas. A falta de interfaces intuitivas e painéis de controle compreensíveis dificulta a visualização dos gastos, limitando o monitoramento eficaz e a transparência no uso dos recursos públicos.

        **3. Desconhecimento da Despesa - Falta de Clareza sobre Credores:** Devido à complexidade dos sistemas contábeis e à grande quantidade de despesas, os Controles Internos frequentemente não têm uma visão clara sobre seus credores. Essa falta de clareza dificulta a análise financeira detalhada e a identificação de relações significativas entre credores e despesas, o que pode comprometer a eficiência na gestão de contratos e nas negociações.

        **4. Dificuldade na Identificação de Padrões e Anomalias:** A análise manual ou mediante relatórios complexos referentes a grandes volumes de dados financeiros é propensa a erros e dificulta a detecção de padrões suspeitos ou anomalias, que podem indicar fraudes, imperícia ou má gestão dos recursos públicos.

        **5. Monitoramento Ineficaz da Sazonalidade e Tendências de Despesas:** A falta de ferramentas para análise de padrões de pagamento e previsão de tendências limita a capacidade da alta gestão de planejar de maneira precisa. Isso resulta em decisões de alocação de recursos subótimas e potencial desperdício de recursos.

        **6. Previsão Inadequada de Necessidades Orçamentárias:** A falta de ferramentas preditivas baseadas no histórico de empenhos dificulta a alocação eficiente de recursos. Isso pode resultar em déficits ou superávits expressivos, que poderiam ser evitados com um planejamento orçamentário mais preciso.
    """)
    st.write("""
    ####  Objetivos da Aplicação

    Com base nos problemas identificados, a aplicação propõe as seguintes soluções para melhorar a controle e o monitoramento das finanças públicas, simplificando o processo, economizando tempo e aumentando a eficiência dos Controles Internos:

    1. **Implementar Análises Simples e Rápidas dos Gastos Efetuados:**  
    Desenvolver um dashboard interativo que permita aos Controles Internos visualizar facilmente os gastos públicos. Isso facilitará o monitoramento eficiente e aumentará a transparência no uso dos recursos públicos, reduzindo a complexidade dos sistemas atuais e economizando tempo.

    2. **Classificação Automática de Credores:**  
    Utilizar algoritmos de aprendizado de máquina para categorizar credores com base em seus padrões de pagamento. Essa abordagem permitirá uma análise financeira mais detalhada e auxiliará na identificação de relações significativas entre credores e despesas, otimizando a gestão de contratos e negociações, de maneira mais simples e direta.

    3. **Detecção de Padrões e Anomalias:**  
    Integrar técnicas avançadas de análise de dados para identificar automaticamente padrões suspeitos e anomalias nos registros financeiros. Isso ajudará na detecção precoce de fraudes ou má gestão dos recursos públicos, aumentando a segurança e a confiabilidade das finanças, enquanto simplifica os processos complexos atuais.

    4. **Análise de Sazonalidade e Tendências:**  
    Desenvolver funcionalidades para analisar padrões de pagamento ao longo do tempo e prever tendências sazonais. Isso possibilitará a tomada de decisões mais informadas e eficazes sobre a alocação de recursos, utilizando uma ferramenta mais intuitiva e de fácil uso.

    5. **Previsão de Necessidades Orçamentárias:**  
    Aplicar modelos preditivos baseados no histórico de empenhos para antecipar as necessidades orçamentárias futuras. Essa funcionalidade ajudará a alta gestão a planejar orçamentos com maior precisão, evitando déficits e superávits imprevistos e garantindo uma alocação de recursos mais eficiente e estratégica, simplificando o planejamento orçamentário tradicionalmente complexo.
    """)
    
    st.write("#### Links úteis (Referências):")
    st.write("""
        - [Portal da Transparência - Governo Federal](https://portaldatransparencia.gov.br/despesas)
        - [Radar da Transparência - Atricon](https://radardatransparencia.atricon.org.br/panel.html)
        - [IEGM - IRB Contas](https://iegm.irbcontas.org.br/)
    """)
    show_dataset()

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

def main():
    """Main function to set up the app."""
    load_css()

    # Sidebar menu using option_menu
    with st.sidebar:
        selected_page = option_menu(
            None,  # Menu title
            ["Home", "About", "Contact"],  # Menu options including Home
            icons=["house", "info-circle", "envelope"],  # Icons for each option
            menu_icon="cast",  # Icon for the menu
            default_index=0,  # Default selected option
            styles={
                "nav-link-selected": {"background-color": "#6342e9"},  # Change selected option color
                "nav-link": {"--hover-color": "#eee"}  # Optional: change hover color
            },
        )

    # Dynamic content loading based on the selected menu option
    if selected_page == "Home":
        home_page()
    elif selected_page == "About":
        
        load_page("About")
    elif selected_page == "Contact":
        load_page("Contact")

if __name__ == "__main__":
    show_banner()
    main()
