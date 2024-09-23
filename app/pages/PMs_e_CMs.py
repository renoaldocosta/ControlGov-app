import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pygwalker.api.streamlit import StreamlitRenderer
from st_aggrid import AgGrid
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# Imports locais
from app.services.text_functions import mkd_text_divider, mkd_text, mkd_paragraph

# Dicionário para traduzir os meses para português
MONTH_TRANSLATION = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
}

def format_df(df):
    """
    Formata o DataFrame removendo colunas indesejadas, formatando valores e datas.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame formatado.
    """
    df = df.drop(columns=['_id'], errors='ignore')
    df['Credor'] = df['Credor'].str.split(' - ').apply(
        lambda x: f'{x[1]} - {x[0]}' if len(x) > 1 else x[0]
    )

    # Remover o símbolo de moeda 'R$' e os pontos dos milhares
    value_columns = ['Alteração', 'Empenhado', 'Liquidado', 'Pago']
    for column in value_columns:
        df[column] = (
            df[column]
            .str.replace(r'R\$ ?', '', regex=True)
            .str.replace('.', '')
            .str.replace(',', '.')
            .astype(float)
        )

    data_columns = ['Data', 'Atualizado']
    for column in data_columns:
        df[column] = pd.to_datetime(df[column])

    return df

def format_currency(value):
    """
    Formata um valor numérico para o estilo brasileiro de moeda.

    Args:
        value (float): Valor numérico.

    Returns:
        str: Valor formatado como moeda brasileira.
    """
    if value != '':
        return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    else:
        return "R$ 0,00"

def metrics(df):
    """
    Exibe métricas do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame para cálculo das métricas.
    """
    if 'df_escolhido' in st.session_state:
        df_escolhido = st.session_state['df_escolhido']
        mkd_text_divider(f"Métricas de {df_escolhido}", level='subheader', position='center')

    # Converter colunas de data para o tipo datetime
    df['data'] = pd.to_datetime(df['data'])
    df['dataEmpenho'] = pd.to_datetime(df['dataEmpenho'])

    # Cálculo das métricas
    total_registros = df.shape[0]
    data_mais_recente = df['data'].max().strftime('%d/%m/%Y')  # Converte a data para string
    data_mais_antiga = df['data'].min().strftime('%d/%m/%Y')    # Converte a data para string
    valor_minimo = df['valor'].min()
    valor_medio = df['valor'].mean()
    valor_maximo = df['valor'].max()

    # Exibir métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", total_registros)
    with col2:
        st.metric("Data Mais Antiga", data_mais_antiga)
    with col3:
        st.metric("Data Mais Recente", data_mais_recente)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Valor Mínimo", format_currency(valor_minimo))
    with col5:
        st.metric("Valor Médio", format_currency(valor_medio))
    with col6:
        st.metric("Valor Máximo", format_currency(valor_maximo))

def year_filter(df):
    """
    Aplica filtro por ano no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame filtrado por ano.
    """
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    first_year = df['data'].dt.year.min()
    last_year = df['data'].dt.year.max()

    if first_year != last_year:
        selected_years = st.slider("Ano", min_value=first_year, max_value=last_year, value=(first_year, last_year))
        df = df[df['data'].dt.year.between(*selected_years)]
    else:
        st.caption(f'Ano selecionado: {first_year}')
        selected_years = [first_year]

    st.session_state['selected_years'] = selected_years
    return df

def month_filter(df):
    """
    Aplica filtro por mês no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame filtrado por mês.
    """
    df['Mês_Numero'] = df['data'].dt.month
    df['Mês'] = df['Mês_Numero'].map(MONTH_TRANSLATION)
    all_months = list(MONTH_TRANSLATION.values())

    months = st.multiselect("Mês", all_months, placeholder="Selecione um ou mais meses")
    if months:
        df = df[df['Mês'].isin(months)]

    df = df.sort_values(by='Mês_Numero')
    st.session_state['selected_months'] = months

    return df

def credores_filter(df):
    """
    Aplica filtro por credores no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame filtrado por credores.
    """
    credores = list(df['credor'].unique())
    selected_creditors = st.multiselect("Credores / Fornecedores", credores, placeholder="Selecione um ou mais credores")

    if selected_creditors:
        df = df[df['credor'].isin(selected_creditors)]
        st.session_state['credor'] = selected_creditors

    return df

def orgao_filter(df):
    """
    Aplica filtro por órgão no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame filtrado por órgão.
    """
    orgaos = list(df['orgao'].unique())
    selected_orgaos = st.multiselect("Órgão", orgaos, placeholder="Selecione um ou mais órgãos")

    if selected_orgaos:
        df = df[df['orgao'].isin(selected_orgaos)]
        st.session_state['orgao'] = selected_orgaos

    return df

def converter_valor_brasileiro(valor_str):
    """
    Converte uma string de valor monetário brasileiro para float.

    Args:
        valor_str (str): Valor monetário em formato brasileiro.

    Returns:
        float or None: Valor numérico ou None se não for possível converter.
    """
    valor_str = valor_str.replace('.', '').replace(',', '.')
    try:
        return float(valor_str)
    except ValueError:
        st.error("O valor inserido não é válido. Verifique o formato.")
        return None

def valores_filter(df):
    """
    Aplica filtro por valor no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame filtrado por valor.
    """
    allow_filter_value = st.checkbox("Filtrar dados por valor", value=False)
    if not allow_filter_value:
        return df
    else:
        tipo = st.radio("Tipo de Filtro", ['Menor', 'Igual', 'Maior'], horizontal=True)
        valor_colado = st.text_input("Valor", value="0,00")
        if valor_colado == '':
            st.warning("Deixar o valor em branco aplicará o filtro com o valor padrão igual a 0,00")
            valor_colado = "0,00"

        filter_value = converter_valor_brasileiro(valor_colado)
        if filter_value is not None:
            if tipo == 'Menor':
                df_filtrado = df[df['valor'] < filter_value]
            elif tipo == 'Igual':
                df_filtrado = df[df['valor'] == filter_value]
            else:
                df_filtrado = df[df['valor'] > filter_value]

            if df_filtrado.empty:
                st.warning("Nenhum registro encontrado para o filtro selecionado. Filtro não aplicado.")
                return df
            else:
                return df_filtrado
        else:
            return df

def filters(df_empenho, df_liquidacao, df_pagamento):
    """
    Aplica filtros no DataFrame selecionado.

    Args:
        df_empenho (pd.DataFrame): DataFrame de empenhos.
        df_liquidacao (pd.DataFrame): DataFrame de liquidações.
        df_pagamento (pd.DataFrame): DataFrame de pagamentos.

    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    with st.sidebar:
        st.markdown("## Filtros")
        base = st.radio("Selecione a base de dados", ['Empenhos', 'Liquidação', 'Pagamento'])
        if base == 'Empenhos':
            df = df_empenho
            st.session_state['df_escolhido'] = 'Empenho'
        elif base == 'Liquidação':
            df = df_liquidacao
            st.session_state['df_escolhido'] = 'Liquidação'
        else:
            df = df_pagamento
            st.session_state['df_escolhido'] = 'Pagamento'

        df = year_filter(df)
        df = month_filter(df)
        df = credores_filter(df)
        df = orgao_filter(df)
        df = valores_filter(df)

        return df

@st.cache_data
def load_data(file, file_type):
    """
    Carrega dados de um arquivo.

    Args:
        file (UploadedFile): Arquivo carregado pelo usuário.
        file_type (str): Tipo do arquivo ('json' ou 'csv').

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.
    """
    if file_type == 'json':
        return pd.read_json(file)
    elif file_type == 'csv':
        return pd.read_csv(file)

def rename_columns(df):
    """
    Renomeia colunas do DataFrame para nomes mais amigáveis.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame com colunas renomeadas.
    """
    novos_nomes_colunas = {
        'NumeroDocumeto': 'Número do Documento',
        'empenho': 'Empenho',
        'data': 'Data',
        'dataEmpenho': 'Data do Empenho',
        'municipio': 'Município',
        'unidade': 'Unidade',
        'cd_Unidade': 'Código da Unidade',
        'orgao': 'Órgão',
        'unidadeOrcamentaria': 'Unidade Orçamentária',
        'credor': 'Credor',
        'valor': 'Valor',
        'cd_UnidadeOrcamentaria': 'Código Unidade Orçamentária',
        'cd_Orgao': 'Código Órgão',
        'cd_ElementoGestor': 'Código Elemento Gestor',
        'de_ElementoGestor': 'Descrição Elemento Gestor',
        'cd_Fonte': 'Código Fonte',
        'de_Fonte': 'Descrição Fonte',
        'Mês_Numero': 'Número do Mês',
        'Mês': 'Mês'
    }
    df.rename(columns=novos_nomes_colunas, inplace=True)
    return df

def remove_columns(df, columns):
    """
    Remove colunas indesejadas do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.
        columns (list): Lista de colunas a serem removidas.

    Returns:
        pd.DataFrame: DataFrame sem as colunas especificadas.
    """
    return df.drop(columns=columns, errors='ignore')

def gerar_botoes_download(df, nome_arquivo_csv, nome_arquivo_json):
    # Converte o DataFrame para CSV
    csv = df.to_csv(index=False)
    
    # Converte o DataFrame para JSON
    json = df.to_json(orient='records')

    # Cria duas colunas
    col1, col2 = st.columns(2)

    # Botão para download do CSV
    with col1:
        st.download_button(
            label="Baixar dados em CSV",
            data=csv,
            file_name=nome_arquivo_csv,
            mime='text/csv',
            use_container_width=True
        )

    # Botão para download do JSON
    with col2:
        st.download_button(
            label="Baixar dados em JSON",
            data=json,
            file_name=nome_arquivo_json,
            mime='application/json',
            use_container_width=True
        )

def gerar_nuvem_de_palavras(df, coluna):
    """Gera uma nuvem de palavras a partir de uma coluna de um DataFrame"""
    
    mkd_text("", level='h7', position='center')
    mkd_text_divider("Nuvem de Palavras", level='subheader', position='center')
    mkd_text(f"{coluna}:", level='h4', position='center')

    # Verificar se a coluna existe no DataFrame
    if coluna not in df.columns:
        st.error(f"A coluna '{coluna}' não foi encontrada no dataframe.")
        return

    # Extrair a coluna e remover valores nulos
    elementos = df[coluna].dropna()

    # Contar a frequência de cada elemento (tratando frases como unidades únicas)
    frequencia = elementos.value_counts().to_dict()

    # Definir stopwords (palavras a serem excluídas)
    stopwords = set(STOPWORDS)
    
    # Opções interativas na barra lateral
    st.sidebar.header("Configurações da Nuvem de Palavras")
    background_color = st.sidebar.selectbox(
        "Selecione a cor de fundo",
        options=["white", "black", "blue", "green", "red", "grey", "yellow"],
        index=0
    )

    max_words = st.sidebar.slider(
        "Número máximo de palavras",
        min_value=10,
        max_value=500,
        value=200,
        step=10
    )

    colormap = st.sidebar.selectbox(
        "Selecione o colormap",
        options=[
            "viridis", "plasma", "inferno", "magma", "cividis",
            "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys",
            "Oranges", "OrRd", "PuBu", "PuBuGn", "PuRd", "Purples",
            "RdPu", "Reds", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd"
        ],
        index=0
    )

    # Gerar a nuvem de palavras
    wordcloud = WordCloud(
        background_color=background_color,
        stopwords=stopwords,
        max_words=max_words,
        colormap=colormap,
        width=800,
        height=400,
        collocations=False,  # Impede que combinações de palavras sejam divididas
        random_state=42
    ).generate_from_frequencies(frequencia)

    # Exibir a nuvem de palavras usando matplotlib
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)

    st.pyplot(fig)
    
    
def run():
    """
    Função principal que executa a aplicação Streamlit.
    """
    mkd_text("Prefeituras e Câmaras/BA", level='title', position='center')
    mkd_text("", level='h7', position='center')

    if 'dados_carregados' not in st.session_state:
        with st.expander("Carregar dados"):
            st.write('Para utilizar o módulo de análise, é necessário carregar os dados: Empenho, Liquidação e Pagamento. Favor carregar cada um dos arquivos separadamente.')
            col1, col2, col3 = st.columns(3)
            with col1:
                file_upload_empenho = st.file_uploader("Carregar Dados de Empenho", type=['csv', 'json'])
            with col2:
                file_upload_liquidacao = st.file_uploader("Carregar Dados de Liquidação", type=['csv', 'json'])
            with col3:
                file_upload_pagamento = st.file_uploader("Carregar Dados de Pagamento", type=['csv', 'json'])

            if file_upload_empenho and file_upload_liquidacao and file_upload_pagamento:
                if st.button("Carregar Dados", use_container_width=True):
                    df_empenho = load_data(file_upload_empenho, 'json' if file_upload_empenho.name.endswith('.json') else 'csv')
                    st.session_state['df_empenho'] = df_empenho

                    df_liquidacao = load_data(file_upload_liquidacao, 'json' if file_upload_liquidacao.name.endswith('.json') else 'csv')
                    st.session_state['df_liquidacao'] = df_liquidacao

                    df_pagamento = load_data(file_upload_pagamento, 'json' if file_upload_pagamento.name.endswith('.json') else 'csv')
                    st.session_state['df_pagamento'] = df_pagamento

                    st.session_state['dados_carregados'] = True
    else:   
        df_empenho = st.session_state['df_empenho']
        df_liquidacao = st.session_state['df_liquidacao']
        df_pagamento = st.session_state['df_pagamento']
        df = filters(df_empenho, df_liquidacao, df_pagamento)
        metrics(df)
        mkd_text("", level='h7', position='center')

        mkd_text_divider("Dados Brutos", level='subheader', position='center')
        df_to_show = df.copy()
        df_to_show = rename_columns(df_to_show)
        columns_to_remove = ['Unidade', 'Município', 'Mês', 'Número do Mês', 'Código Órgão', 'Código da Unidade']
        df_to_show = remove_columns(df_to_show, columns_to_remove)
        AgGrid(df_to_show, height=300, width='100%')
        gerar_botoes_download(df_to_show, 'dados.csv', 'dados.json')
        gerar_nuvem_de_palavras(df_to_show, 'Descrição Elemento Gestor')
        #st.write(df_to_show.columns)

if __name__ == "__main__":
    st.session_state.clear()
    run()
        
