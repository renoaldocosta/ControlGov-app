import streamlit as st
import os

# Função para criar caixas estilizadas com tons de azul
def styled_box(header, content, header_bg_color, content_bg_color):
    st.markdown(
        f"""
        <div style="background-color: {header_bg_color}; padding: 10px; border-radius: 5px; color: white; font-weight: bold; text-align: center; font-size: 18px;">
            {header}
        </div>
        <div style="background-color: {content_bg_color}; padding: 15px; border-radius: 5px; margin-bottom: 10px; color: black; font-size: 16px;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )

def load_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    

def show_file_md(path):
    file_path = path

    if not os.path.exists(file_path):
        st.error(f"Arquivo não encontrado: {file_path}")
        return

    markdown_content = load_markdown_file(file_path)
    st.markdown(markdown_content)
    
    
def run():
    # Título do projeto
    st.title("Business Model Canvas")
    

    # Layout do BMC usando colunas e markdown, com tons de azul
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        styled_box(
            "Segmentos de Usuários", 
            "• Câmaras Municipais do eixo Bahia|Sergipe<br>• Prefeituras Municipais do eixo Bahia|Sergipe<br>• Profissionais de controle interno e gestores públicos", 
            "#1E3D59", "#A9C4D9"
        )
    with col2:
        styled_box(
            "Proposta de Valor", 
            "• Melhorar a eficiência na gestão e monitoramento das finanças públicas<br>• Proporcionar transparência e segurança nos gastos públicos<br>• Permitir previsões orçamentárias precisas<br>• Facilitar a tomada de decisões baseadas em dados", 
            "#1E3D59", "#A9C4D9"
        )
    with col3:
        styled_box(
            "Canais", 
            "• Interface web interativa utilizando Streamlit<br>• Acesso direto via navegadores web em servidores seguros<br>• Relatórios e dashboards disponibilizados para download", 
            "#1E3D59", "#A9C4D9"
        )

    col4, col5 = st.columns([1, 1])
    with col4:
        styled_box(
            "Fontes de Receita", 
            "• Doações<br>• Cobrança por serviços adicionais, como customização de relatórios<br>• Consultoria em análise de dados e treinamento específico", 
            "#1E3D59", "#A9C4D9"
        )
    with col5:
        styled_box(
            "Recursos Principais", 
            "• Dados financeiros dos Portais de Transparência<br>• Infraestrutura de TI: servidores seguros<br>• Ferramentas de ciência de dados: pandas, scikit-learn, TensorFlow<br>• Equipe técnica da área de dados", 
            "#1E3D59", "#A9C4D9"
        )

    col6, col7 = st.columns([1, 1])
    with col6:
        styled_box(
            "Atividades-Chave", 
            "• Coleta e processamento de dados financeiros<br>• Desenvolvimento e manutenção da aplicação web<br>• Implementação de modelos de machine learning<br>• Análise contínua da qualidade dos dados", 
            "#1E3D59", "#A9C4D9"
        )
    with col7:
        styled_box(
            "Parcerias Principais", 
            "• Fornecedores de dados financeiros<br>• Empresas de hospedagem segura<br>• Comunidades e ferramentas open-source<br>• Universidades e institutos de pesquisa", 
            "#1E3D59", "#A9C4D9"
        )

    # Caixa de Estrutura de Custos
    styled_box(
        "Estrutura de Custos", 
        "• Custo com servidores para hospedagem segura<br>• Desenvolvimento e manutenção contínua da aplicação<br>• Treinamento de equipe técnica e usuários<br>• Aquisição de APIs ou serviços de dados", 
        "#1E3D59", "#A9C4D9"
    )

    # Caixa de Indicadores e Metas
    styled_box(
        "Indicadores e Metas", 
        "• Implementar e disponibilizar o dashboard no prazo<br>• Reduzir o tempo de análise financeira em 50%<br>• Precisão de 95% na detecção de anomalias<br>• Melhorar a acurácia das previsões em 80%<br>• Satisfação do usuário de 85%", 
        "#1E3D59", "#A9C4D9"
    )
    
    
    # Definir o diretório base
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Caminho absoluto para Charter.md
    charter_path = os.path.join(base_path, '..', '..', 'Docs', 'Project', 'Charter.md')
    show_file_md(charter_path)

    # Caminho absoluto para DataSummaryReport.md
    data_summary_path = os.path.join(base_path, '..', '..', 'Docs', 'Data_Report', 'DataSummaryReport.md')
    show_file_md(data_summary_path)





if __name__ == "__main__":
    run()