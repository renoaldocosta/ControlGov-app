import os
import time
import requests
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
from typing import Dict, Optional
import plotly.express as px
from app.services.text_functions import mkd_text, mkd_text_divider
import plotly.io as pio



# LangChain
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

# LangChain Community
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI

# LangChain OpenAI
from langchain_openai.chat_models import ChatOpenAI


load_dotenv()

secret = os.getenv("API_SECRET")
openai_api_key = os.getenv("OPENAI_API_KEY")


def consultar_cpf_cnpj(query: str) -> str:
    import requests

    url = "https://api.controlgov.org/embeddings/subelementos"
    payload = {
        "query": query,
        "secret": secret,  # É recomendável armazenar o secret em uma variável de ambiente
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, timeout=10)

    if response.status_code == 200:
        data = response.json()
        return data.get("resposta", "Nenhuma resposta encontrada.")
    else:
        return f"Erro ao consultar a API: {response.status_code}"


def consultar_PessoaFisica_PessoaJuridica(query: str) -> str:
    import requests

    url = "https://api.controlgov.org/embeddings/subelementos"
    payload = {
        "query": query,
        "secret": secret,  # É recomendável armazenar o secret em uma variável de ambiente
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, timeout=10)

    if response.status_code == 200:
        data = response.json()
        return data.get("resposta", "Nenhuma resposta encontrada.")
    else:
        return f"Erro ao consultar a API: {response.status_code}"


def consultar_subelementos(query: str) -> str:
    import requests

    url = "https://api.controlgov.org/embeddings/subelementos"
    payload = {
        "query": query,
        "secret": secret,  # É recomendável armazenar o secret em uma variável de ambiente
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, timeout=10)

    if response.status_code == 200:
        data = response.json()
        data["resposta"] = data["resposta"] + "\nFormatar valores em R$"
        return data.get("resposta", "Nenhuma resposta encontrada.")
    else:
        return f"Erro ao consultar a API: {response.status_code}"


def consultar_empenhado_sum(query=None):
    url = "https://api.controlgov.org/elementos/despesa/empenhado-sum/"
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        elementos = data.get("elementos", [])
        if not elementos:
            return "Nenhum dado encontrado."

        resultado = "Soma dos Valores Empenhados por Elemento de Despesa:\n\n"
        for elemento in elementos:
            elemento_despesa = elemento.get("elemento_de_despesa", "Desconhecido")
            total_empenhado = elemento.get("total_empenhado", 0)
            resultado += f"• {elemento_despesa}: R\$ {total_empenhado:,.2f}\n"

        return resultado

    except requests.exceptions.RequestException as e:
        return f"Ocorreu um erro ao consultar a API: {e}"
    except ValueError:
        return "Erro ao processar a resposta da API."


def listar_empenhos_por_elemento(query=None):
    """
    Consulta os empenhos por elemento de despesa e retorna uma lista formatada.

    Args:
        query (str, opcional): Um termo para filtrar os elementos de despesa.
                               Se None, retorna todos os elementos.

    Returns:
        str: Lista formatada de empenhos por elemento com '\n' ao final de cada linha.
    """
    url = "https://api.controlgov.org/elementos/despesa/empenhado-sum/"
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        elementos = data.get("elementos", [])
        if not elementos:
            return "Nenhum dado encontrado."

        # Se uma consulta específica for fornecida, filtrar os elementos
        if query:
            elementos = [
                elem
                for elem in elementos
                if query.lower() in elem.get("elemento_de_despesa", "").lower()
            ]
            if not elementos:
                return (
                    f"Nenhum empenho encontrado para o elemento de despesa: '{query}'."
                )

        resultado = "Empenhos por Elemento de Despesa:\n\n"
        for elemento in elementos:
            elemento_despesa = elemento.get("elemento_de_despesa", "Desconhecido")
            total_empenhado = elemento.get("total_empenhado", 0)
            resultado += f"• {elemento_despesa}: R\$ {total_empenhado:,.2f}\n"

        return resultado

    except requests.exceptions.RequestException as e:
        return f"Ocorreu um erro ao consultar a API: {e}"
    except ValueError:
        return "Erro ao processar a resposta da API."


def generate_response_agent(text):
    # Consultar CPF ou CNPJ
    text = "Me responda apenas:\n" + text
    consultar_cpf_cnpj_tool = Tool(
        name="Consultar CPF ou CNPJ",
        func=consultar_cpf_cnpj,
        description=(
            "Use esta ferramenta para obter informações sobre CPF ou CNPJ de um credor."
            "Por exemplo, você pode perguntar: 'Qual o CPF do credor <nome> com asteriscos?' ou 'Qual o CNPJ do credor <nome>?'"
        ),
    )

    # Definir as ferramentas
    subelementos_tool = Tool(
        name="Consultar Subelemento Individualmente",
        func=consultar_subelementos,
        description=(
            "Use esta ferramenta para obter informações sobre alguns subelementos financeiros. "
            "Por exemplo, você pode perguntar: 'Qual o total empenhado para o subelmento <subelemento>?'"
        ),
    )

    empenho_pessoa_fisica_juridica = Tool(
        name="Consultar Empenho a Pessoa Física ou Jurídica",
        func=consultar_PessoaFisica_PessoaJuridica,
        description=(
            "Use esta ferramenta para obter informações sobre valores empenhados para Pessoa Física ou Pessoa Jurídica. "
            "Por exemplo, você pode perguntar: 'Qual o total empenhado para <Pessoa Física>?' ou 'Qual o total empenhado para <Pessoa Jurídica>?'"
        ),
    )

    empenhos_por_elemento_tool = Tool(
        name="Consultar Todos os Elementos de uma Vez",
        func=listar_empenhos_por_elemento,
        description=(
            "Use esta ferramenta para obter a lista de empenhos por elemento de despesa. "
            "Por exemplo, você pode perguntar: 'Quais são os empenhos para o elemento de despesa por Obrigação Patronal?' "
        ),
    )
    # "Ou simplesmente: 'Liste todos os empenhos por elemento de despesa.'"

    tools = [
        subelementos_tool,
        # empenhado_sum_tool,
        empenhos_por_elemento_tool,
        empenho_pessoa_fisica_juridica,
        consultar_cpf_cnpj_tool,
        # categoria_tool  # Adiciona a nova ferramenta aqui
    ]

    prefix = """Você é um assistente direto e especializado em finanças governamentais.
    Você pode ajudar os usuários a consultar informações sobre:
    - elementos e subelementos de despesa, 
    - consultas aos valores empenhados a Pessoas Físicas e Jurídicas.
    - consultas a CPF ou CNPJ dos credores.
    
    Você tem acesso às seguintes ferramentas:
    
    - Consultar Empenho a Pessoa Física ou Jurídica: Use esta ferramenta para obter informações sobre valores empenhados para PF ou PJ.
    - Consultar CPF ou CNPJ: Use esta ferramenta para obter informações sobre CPF ou CNPJ dos credores.
    - Consultar Subelemento Individualmente: Use esta ferramenta para obter informações sobre valores empenhados por subelementos de despesa.
    - Consultar Todos os Elementos de uma Vez: Use esta ferramenta para obter a lista de valores empenhados por elemento de despesa.
    
    """
    # - Consultar Empenhado Sum: Use esta ferramenta para obter a soma de todos os valores empenhados para cada elemento de despesa.

    suffix = """
    Histórico do Chat:
    {chat_history}
    Última Pergunta: {input}
    {agent_scratchpad}
    Sempre responda em Português.
    Responda apenas ao que foi perguntado. Evite informações desnecessárias.
    """

    # Atualizar o prefixo do prompt para incluir a nova ferramenta
    prompt = ConversationalAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    # Configurar a memória
    msg = StreamlitChatMessageHistory()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            messages=msg, memory_key="chat_history", return_messages=True
        )
    memory = st.session_state.memory

    # Configurar o LLM
    llm_chain = LLMChain(
        llm=ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini"),
        prompt=prompt,
        verbose=True,
    )

    # Configurar o agente
    agent = ConversationalAgent(
        llm_chain=llm_chain,
        memory=memory,
        verbose=True,
        max_interactions=3,
        tools=tools,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, memory=memory, verbose=True
    )

    # Executar o agente
    if text:
        result = agent_executor.run(text)

        # Verificar se a resposta é a solicitação de mais informações
        if "não consegui identificar" in result.lower():
            return (
                result  # Retorna imediatamente a mensagem solicitando mais informações
            )
        else:
            return result


def response_generation(text: str, openai_api_key):
    # time.sleep(1)
    with st.spinner("Estou pensando..."):
        st.toast("Pensando!", icon="🤖")
        response = (
            generate_response_agent(text)
            .replace("R$ ", "R\$ ")
            .replace(".\n```", "")
            .replace("*", "\*")
        )

    return response
    # with st.chat_message("assistant"):
    #     st.write(response)

    # st.session_state.messages.append({"role": "assistant", "content": response})


def run_chat(openai_api_key: str):
    mkd_text("🤖 Chatbot de Atendimento CMP", level="subheader", position="center")
    column_novo, column_mostrar_chat = st.columns([0.1, 0.9])
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    return column_novo, column_mostrar_chat
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"].replace("R$ ", "R\$ "))

    # if prompt := st.chat_input("O que você deseja consultar?"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})

    #     with st.chat_message("user"):
    #         st.write(prompt)

    #     response = response_generation(prompt, openai_api_key)

    #     st.session_state.messages.append({"role": "assistant", "content": response})

    #     st.toast(response, icon='🤖')
    #     time.sleep(15)

    #     with st.chat_message("assistant"):
    #         st.write(response)


# Dictionary to translate month numbers to Portuguese month names
MONTH_TRANSLATION = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Março",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}


def test_mongodb_connection(db_name: str, collection_name: str) -> MongoClient:
    """
    Test the connection to MongoDB and return the collection.

    Args:
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.

    Returns:
        collection: The MongoDB collection object.

    Raises:
        SystemExit: If unable to connect to the database.
    """
    with st.spinner("Testing MongoDB Connection..."):
        load_dotenv()
        db_password = os.environ.get("db_password")
        uri = f"mongodb+srv://renoaldo_teste:{db_password}@cluster0.zmdkz1p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(uri)

        try:
            client.admin.command("ping")

            db = client[db_name]
            collection = db[collection_name]
            return collection
        except Exception as e:
            st.error(f"Error: {e}")
            raise SystemExit(
                "Unable to connect to the database. Please check your URI."
            )


def format_df(df_empenhos: pd.DataFrame) -> pd.DataFrame:
    """
    Format the 'df_empenhos' DataFrame by cleaning and converting data types.

    Args:
        df_empenhos (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: The formatted DataFrame.
    """
    df = df_empenhos.copy()

    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    if "Credor" in df.columns:
        df["Credor"] = (
            df["Credor"]
            .str.split(" - ")
            .apply(lambda x: f"{x[1]} - {x[0]}" if len(x) > 1 else x[0])
        )

    value_columns = ["Alteração", "Empenhado", "Liquidado", "Pago"]
    for column in value_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .str.replace(r"R\$ ?", "", regex=True)
                .str.replace(".", "")
                .str.replace(",", ".")
            )
            df[column] = df[column].astype(float)
        else:
            st.warning(f"Column '{column}' not found in DataFrame.")

    data_columns = ["Data", "Atualizado"]
    for column in data_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column]).dt.date
        else:
            st.warning(f"Column '{column}' not found in DataFrame.")

    return df


def converte_real_float(df, list_columns):
    for column in list_columns:
        new_column = column + "_float"
        df[new_column] = (
            df[column]
            .str.replace("R\$", "", regex=True)  # Remove currency symbol
            .str.replace(".", "", regex=False)  # Remove thousands separator
            .str.replace(",", ".", regex=False)  # Replace decimal commas with dots
            .astype(float)  # Convert to float
        )
    return df  # Fixed return statement


def converte_data_datetime(df, list_columns):
    for column in list_columns:
        new_column = column + "_datetime"
        df[new_column] = pd.to_datetime(df[column], errors="coerce")
    return df


def processa_itens_column(value):
    if isinstance(value, list):
        # Se for uma lista aninhada, converte cada item interno para string e faz join
        return ", ".join(
            ", ".join(map(str, item)) if isinstance(item, list) else str(item)
            for item in value
        )
    return str(value)  # Caso não seja uma lista, apenas converte para string


def controlgov_api_request(url: str) -> pd.DataFrame:
    """
    Make a request to the ControlGov API and return the data as a DataFrame.

    Args:
        url (str): The URL of the API endpoint.

    Returns:
        pd.DataFrame: The data from the API as a DataFrame.
    """
    with st.spinner("Loading data..."):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data["empenhos"])
            else:
                st.error(f"Erro ao acessar a API: {response.status_code}")
        except Exception as e:
            st.error(f"Erro ao acessar a API: {e}")
            return pd.DataFrame()
    return df


def split_credor_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Credor" in df.columns:
        df["Credor"] = (
            df["Credor"]
            .str.split(" - ")
            .apply(lambda x: f"{x[1]} - {x[0]}" if len(x) > 1 else x[0])
        )
    return df


def extrai_itens_para_colunas(itens):
    """Extrai a descrição e o valor total dos itens."""
    if isinstance(itens, list) and len(itens) > 1:
        try:
            valores = itens[1][0]  # Acessa a lista interna com os valores
            descricao = valores[0]  # Descrição do item
            valor_total = valores[-1]  # Valor total do item
            return descricao, valor_total
        except IndexError:
            # Caso a estrutura dos dados não esteja correta
            return None, None
    return None, None


@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_empenhos_API() -> pd.DataFrame:
    st.session_state.clear()
    with st.spinner("Loading data..."):
        if "df_empenhos" not in st.session_state:
            url = "https://api.controlgov.org/empenhos/"
            df_empenhos = controlgov_api_request(url)
            df_empenhos = df_empenhos.astype(str)
            # Remove the 'id' column
            df_empenhos = df_empenhos.drop(["id"], axis=1)
            # Convert date columns to datetime
            colunas_data = ["Data", "Atualizado"]
            df_empenhos = converte_data_datetime(df_empenhos, colunas_data)
            st.write("")
            # Convert multiple currency columns to float
            colunas_valores = ["Alteração", "Empenhado", "Liquidado", "Pago"]
            df_empenhos = converte_real_float(df_empenhos, colunas_valores)

            # Extrair descrição e valor dos itens
            if "Item(ns)" in df_empenhos.columns:
                df_empenhos[["Item_Descricao", "Item_Valor"]] = df_empenhos[
                    "Item(ns)"
                ].apply(lambda x: pd.Series(extrai_itens_para_colunas(x)))
                df_empenhos = df_empenhos.drop(
                    columns=["Item(ns)"]
                )  # Remove coluna antiga

            df_empenhos = split_credor_column(df_empenhos)

            st.session_state["df_empenhos"] = df_empenhos

            # renomear colunas
            # df_empenhos.rename(columns={'Elemento_de_Despesa': 'Elemento de Despesa','Categorias_de_base_legal':'Categorias de base legal'}, inplace=True)
            return df_empenhos
        else:
            return st.session_state["df_empenhos"]


@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_empenhos(db_name: str, collection_name: str) -> pd.DataFrame:
    """
    Retrieve 'empenhos' data from MongoDB and return it as a DataFrame.

    Args:
        db_name (str): The name of the MongoDB database.
        collection_name (str): The name of the collection.

    Returns:
        pd.DataFrame: The 'empenhos' data as a DataFrame.
    """
    with st.spinner("Loading data..."):
        mongodb_collection = test_mongodb_connection(db_name, collection_name)
        if "df_empenhos" not in st.session_state:
            df_empenhos = pd.DataFrame(list(mongodb_collection.find()))

            if "Item(ns)" in df_empenhos.columns:
                df_empenhos["Item(ns)"] = df_empenhos["Item(ns)"].apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )

            df_empenhos = format_df(df_empenhos)
            return df_empenhos
        else:
            return st.session_state["df_empenhos"]


def metrics(df: pd.DataFrame):
    """
    Display metrics of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame for which to calculate metrics.
    """
    if "df_escolhido" in st.session_state:
        df_escolhido = st.session_state["df_escolhido"]
        mkd_text_divider(
            f"Métricas de {df_escolhido}", level="subheader", position="center"
        )
    else:
        mkd_text_divider("Métricas de Empenho", level="subheader", position="center")

    # if df['Data_datetime'].dtype != 'datetime64[ns]':
    #     df['Data_datetime'] = pd.to_datetime(df['Data'])

    total_registros = df.shape[0]
    data_mais_recente = df["Data_datetime"].max().strftime("%d/%m/%Y")
    data_mais_antiga = df["Data_datetime"].min().strftime("%d/%m/%Y")

    valor_minimo = df["Empenhado_float"].min()
    valor_medio = df["Empenhado_float"].mean()
    valor_maximo = df["Empenhado_float"].max()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", total_registros)
    with col2:
        st.metric("Data Mais Antiga", data_mais_antiga)
    with col3:
        st.metric("Data Mais Recente", data_mais_recente)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Valor Mínimo", format_currency(valor_minimo, currency_symbol="R$"))
    with col5:
        st.metric("Valor Médio", format_currency(valor_medio, currency_symbol="R$"))
    with col6:
        st.metric("Valor Máximo", format_currency(valor_maximo, currency_symbol="R$"))


def format_currency(value: float, currency_symbol: str = "") -> str:
    """
    Format a numeric value to Brazilian currency format.

    Args:
        value (float): Numeric value.
        currency_symbol (str, optional): Currency symbol to prepend. Defaults to ''.

    Returns:
        str: Value formatted as Brazilian currency.
    """
    if value != "" and not pd.isnull(value):
        formatted_value = (
            f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        if currency_symbol:
            return f"{currency_symbol} {formatted_value}"
        else:
            return formatted_value
    else:
        if currency_symbol:
            return f"{currency_symbol} 0,00"
        else:
            return "0,00"


def apply_currency_format(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply Brazilian currency formatting to specific columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the columns to be formatted.
        columns (list): List of columns to apply currency formatting.

    Returns:
        pd.DataFrame: DataFrame with specified columns formatted as Brazilian currency.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(format_currency)
        else:
            st.warning(f"The column '{column}' is not present in the DataFrame.")
    return df


def year_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on a selected range of years.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    first_year = df["Data_datetime"].dt.year.min()
    last_year = df["Data_datetime"].dt.year.max()

    selected_years = st.slider(
        "Ano",
        min_value=int(first_year),
        max_value=int(last_year),
        value=(int(first_year), int(last_year)),
    )

    df_filtered = df[df["Data_datetime"].dt.year.between(*selected_years)]
    st.session_state["selected_years"] = selected_years

    return df_filtered


def month_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected months.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df["Mês_Numero"] = df["Data_datetime"].dt.month
    df["Mês"] = df["Mês_Numero"].map(MONTH_TRANSLATION)
    all_months = list(MONTH_TRANSLATION.values())

    months = st.multiselect("Mês", all_months, placeholder="Selecione um ou mais meses")

    if months:
        df = df[df["Mês"].isin(months)]

    df = df.sort_values(by="Mês_Numero")
    st.session_state["selected_months"] = months

    return df


def credores_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected creditors.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    credores_list = df["Credor"].unique().tolist()

    selected_creditors = st.multiselect(
        "Credores / Fornecedores",
        credores_list,
        placeholder="Selecione um ou mais credores",
    )

    if selected_creditors:
        df = df[df["Credor"].isin(selected_creditors)]

    st.session_state["credores"] = df

    return df


def elemento_despesa_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected 'Elemento de Despesa'.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    elementos = df["Elemento_de_Despesa"].unique().tolist()
    selected_elementos: list[str] = st.multiselect(
        "Elemento de Despesa",
        elementos,
        placeholder="Selecione um ou mais elementos de despesa",
    )

    if selected_elementos:
        df = df[df["Elemento_de_Despesa"].isin(selected_elementos)]

    st.session_state["elemento_despesa"] = df

    return df


def sub_elemento_despesa_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected 'Subelemento'.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    sub_elementos = sorted(df["Subelemento"].unique().tolist())

    selected_sub_elementos = st.multiselect(
        "Subelemento de Despesa",
        sub_elementos,
        placeholder="Selecione um ou mais subelementos de despesa",
    )

    if selected_sub_elementos:
        df = df[df["Subelemento"].isin(selected_sub_elementos)]

    st.session_state["sub_elemento_despesa"] = df

    return df


def categorias_de_base_legal_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected 'Categorias de base legal'.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    base_legal_list = sorted(df["Categorias_de_base_legal"].unique().tolist())

    selected_base_legal = st.multiselect(
        "Base Legal", base_legal_list, placeholder="Selecione uma ou mais bases legais"
    )

    if selected_base_legal:
        df = df[df["Categorias_de_base_legal"].isin(selected_base_legal)]

    st.session_state["base_legal"] = df

    return df


def filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multiple filters to the DataFrame based on user selections.

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    with st.sidebar:
        try:
            with st.expander("## Filtros", expanded=True):
                level = "h4"
                mkd_text("Período", level=level, position="center")
                df = year_filter(df)
                df = month_filter(df)

                mkd_text("Outros Filtros", level=level, position="center")
                df = credores_filter(df)
                df = elemento_despesa_filter(df)
                df = sub_elemento_despesa_filter(df)
                df = categorias_de_base_legal_filter(df)
                df = df.sort_values(by="Data_datetime", ascending=False)
                if "choice_grid" in st.session_state:
                    choice_grid = st.session_state["choice_grid"]
                else:
                    choice_grid = st.radio(
                        "Exibição da Tabela",
                        ["Estilo 1", "Estilo 2"],
                        index=1,
                        horizontal=True,
                    )
                return df, choice_grid
        except Exception as e:
            st.error(f"Erro ao aplicar filtros: {e}")
            return df


def remove_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove unwanted columns from the DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame.
        columns (list): List of columns to be removed.

    Returns:
        pd.DataFrame: DataFrame without the specified columns.
    """
    return df.drop(columns=columns, errors="ignore")


def display_aggrid_with_links(
    df: pd.DataFrame,
    link_columns: list,
    link_text: Optional[Dict[str, str]] = None,
    right_align_columns: list = [],
    height: int = 300,
    theme: str = "balham",
    update_mode: GridUpdateMode = GridUpdateMode.NO_UPDATE,
):
    """
    Display a DataFrame in AgGrid with specified columns rendered as clickable links and custom alignment.

    Args:
        df (pd.DataFrame): The DataFrame to display.
        link_columns (list): List of column names that contain URLs to be rendered as links.
        link_text (dict, optional): Dictionary where keys are link column names and values are link display texts.
                                    If not provided, the default text will be 'Detalhes'.
        right_align_columns (list, optional): List of column names to be right-aligned.
        height (int, optional): Height of the AgGrid table. Defaults to 300.
        theme (str, optional): Theme for AgGrid. Defaults to 'balham'.
        update_mode (GridUpdateMode, optional): Grid update mode. Defaults to GridUpdateMode.NO_UPDATE.
    """
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, flex=1)

    for col in link_columns:
        current_link_text = link_text.get(col, "Detalhes") if link_text else "Detalhes"

        cell_renderer = JsCode(f"""
            class UrlCellRenderer {{
                init(params) {{
                    this.eGui = document.createElement('a');
                    this.eGui.innerText = '{current_link_text}';
                    this.eGui.setAttribute('href', params.value);
                    this.eGui.setAttribute('target', '_blank');
                    this.eGui.style.color = '#1a0dab';
                    this.eGui.style.textDecoration = 'none';
                }}
                getGui() {{
                    return this.eGui;
                }}
            }}
        """)

        gb.configure_column(
            col,
            headerName=col.replace("_", " ").title(),
            cellRenderer=cell_renderer,
            width=150,
            suppressSizeToFit=True,
        )

    if right_align_columns:
        for col in right_align_columns:
            gb.configure_column(
                col,
                headerName=col.replace("_", " ").title(),
                cellStyle={"text-align": "right"},
                type=["numericColumn"],
            )

    gridOptions = gb.build()

    AgGrid(
        df,
        gridOptions=gridOptions,
        update_mode=update_mode,
        height=height,
        theme=theme,
        allow_unsafe_jscode=True,
    )


def display_dataframe_with_links(
    df: pd.DataFrame,
    link_columns: list,
    link_text: dict = None,
    right_align_columns: list = None,
    height: int = 300,
    theme: str = "default",
    use_data_editor: bool = False,
):
    """
    Display a DataFrame in Streamlit with specified columns rendered as clickable links and custom alignment.

    Args:
        df (pd.DataFrame): The DataFrame to display.
        link_columns (list): List of column names that contain URLs to be rendered as links.
        link_text (dict, optional): Dictionary where keys are link column names and values are link display texts.
                                    If not provided, the default text will be the full URL.
        right_align_columns (list, optional): List of column names to be right-aligned.
        height (int, optional): Height of the table in Streamlit. Defaults to 300.
        theme (str, optional): Theme for the component (for data_editor). Defaults to 'default'.
        use_data_editor (bool, optional): If True, uses st.data_editor; otherwise, uses st.dataframe. Defaults to False.
    """
    column_config = {}

    for col in link_columns:
        column_config[col] = st.column_config.LinkColumn(
            label=col.replace("_", " ").title(),
            display_text=link_text.get(col, None) if link_text else None,
            width="large",
            help=f"Link para {col.replace('_', ' ').lower()}",
            validate=r"^https?://.+$",
        )

    if right_align_columns:
        for col in right_align_columns:
            column_config[col] = st.column_config.NumberColumn(
                label=col.replace("_", " ").title(),
                format="%.2f",
                width="small",
                help=f"Valores para {col.replace('_', ' ').lower()}",
            )

    if use_data_editor:
        st.data_editor(
            df,
            column_config=column_config,
            hide_index=True,
            height=height,
            use_container_width=True,
        )
    else:
        st.dataframe(
            df,
            column_config=column_config,
            hide_index=True,
            height=height,
            use_container_width=True,
        )


def reorder_columns(df: pd.DataFrame, new_order: list) -> pd.DataFrame:
    """
    Reorder the columns of a DataFrame based on a provided new order.

    Args:
        df (pd.DataFrame): The original DataFrame.
        new_order (list): List specifying the new order of columns.

    Returns:
        pd.DataFrame: DataFrame with columns reordered.
    """
    missing_cols = [col for col in new_order if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following columns are missing in the DataFrame: {missing_cols}"
        )

    return df[new_order]


def format_data_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Format date columns in the DataFrame to 'dd/mm/yyyy' string format.

    Args:
        df (pd.DataFrame): DataFrame containing date columns.
        columns (list): List of date columns to format.

    Returns:
        pd.DataFrame: DataFrame with formatted date columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%d/%m/%Y")
        else:
            st.warning(f"Column '{col}' not found in DataFrame.")
    return df


def prepare_dataframe(df):
    """
    Prepara o DataFrame para exibição, removendo colunas desnecessárias,
    reordenando e renomeando colunas.
    """
    # Remover colunas indesejadas
    columns_to_remove = [
        "Poder",
        "Função",
        "Subfunção",
        "Item(ns)",
        "Mês",
        "Mês_Numero",
        "Unid. Administradora",
        "Unid. Orçamentária",
        "Fonte de recurso",
    ]
    df = remove_columns(df, columns_to_remove)

    # Reordenar colunas
    new_column_order = [
        "Número",
        "Data",
        "Subelemento",
        "Credor",
        "Alteração",
        "Empenhado",
        "Liquidado",
        "Pago",
        "Atualizado",
        "link_Detalhes",
        "Elemento_de_Despesa",
        "Projeto_Atividade",
        "Categorias_de_base_legal",
        "Histórico",
    ]
    df = reorder_columns(df, new_column_order)

    # Renomear colunas
    colunas_renomeadas = {
        "Número": "Número do Empenho",
        "Data": "Data do Empenho",
        "Subelemento": "Subelemento de Despesa",
        "Credor": "Credor",
        "Alteração": "Alteração no Empenho",
        "Empenhado": "Valor Empenhado",
        "Liquidado": "Valor Liquidado",
        "Pago": "Valor Pago",
        "Atualizado": "Última Atualização",
        "link_Detalhes": "Link para Detalhes",
        "Elemento_de_Despesa": "Elemento de Despesa",
        "Projeto_Atividade": "Projeto / Atividade",
        "Categorias_de_base_legal": "Categoria Legal",
        "Histórico": "Descrição do Histórico",
    }
    df = df.rename(columns=colunas_renomeadas)

    df = df.reset_index(drop=True)

    return df


def display_data(df):
    """
    Exibe o DataFrame com formatação apropriada no aplicativo Streamlit.
    """
    # Definir colunas para links e alinhamento
    link_cols = ["Link para Detalhes"]
    link_texts = {"Link para Detalhes": "Ver Detalhes"}
    right_align_cols = [
        "Valor Empenhado",
        "Valor Liquidado",
        "Valor Pago",
        "Alteração no Empenho",
    ]

    # Exibir DataFrame com links e formatação
    display_aggrid_with_links(
        df=df,
        link_columns=link_cols,
        link_text=link_texts,
        right_align_columns=right_align_cols,
        height=300,
        theme="balham",
    )


def run():
    """
    Função principal para executar o aplicativo Streamlit.
    """
    # Exibir o título
    mkd_text("Câmara Municipal de Pinhão - SE", level="title", position="center")

    # Obter dados
    df_empenhos = get_empenhos_API()
    # st.dataframe(df_empenhos)

    # Filtrar dados
    df_filtered, choice_grid = filters(df_empenhos)

    # Exibir métricas
    metrics(df_filtered)

    # Divisor de texto
    mkd_text("", level="subheader", position="center")
    mkd_text_divider("Registros", level="subheader", position="center")

    # Criar abas
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Empenhos", "Liquidações", "Pagamentos", "**ChatBot** 🤖"]
    )

    # Preparar o DataFrame para exibição
    df_to_show = prepare_dataframe(df_filtered)

    # Exibir dados na primeira aba
    with tab1:
        if choice_grid == "Estilo 2":
            display_data(df_to_show)
        else:
            st.dataframe(df_to_show)

        mkd_text_divider("Visualizações", level="subheader", position="center")
        visualizacoes(df_filtered)

    # Segunda aba (Exploração)
    with tab4:
        st.write()
        column_novo, column_mostrar_chat = run_chat(openai_api_key)

        conteiner_chat = st.container()

        if "atualizar_chat" not in st.session_state:
            st.session_state.atualizar_chat = False
        if "cont_chat" not in st.session_state:
            st.session_state.cont_chat = 0
        # st.write(st.session_state.atualizar_chat)
        # st.write(st.session_state.cont_chat)
        # st.write(len(st.session_state.messages))
        if "atualizar_chat" in st.session_state:
            if st.session_state.atualizar_chat:
                if st.session_state.atualizar_chat:
                    cont = 0
                    with conteiner_chat:
                        for message in st.session_state.messages_backup:
                            with st.chat_message(message["role"]):
                                st.write(message["content"].replace("R$ ", "R\$ "))
                            cont += 1
                    st.session_state.cont_chat = cont
                    st.session_state.atualizar_chat = False
        with column_novo:
            if st.button("Novo", use_container_width=True):
                st.session_state.messages.clear()
                st.session_state.messages_backup.clear()
                st.session_state.cont_chat = 0
                st.session_state.atualizar_chat = False
        if not st.session_state.atualizar_chat or st.session_state.cont_chat != len(
            st.session_state.messages
        ):
            with column_mostrar_chat:
                if st.button("Mostrar Chat Atualizado", use_container_width=True):
                    # st.write(st.session_state.atualizar_chat)
                    # st.write(st.session_state.cont_chat)
                    # st.write(len(st.session_state.messages))
                    st.session_state.atualizar_chat = True
                    st.session_state.messages_backup = st.session_state.messages

                    if "atualizar_chat" in st.session_state:
                        if st.session_state.atualizar_chat:
                            cont = 0
                            with conteiner_chat:
                                for message in st.session_state.messages_backup:
                                    with st.chat_message(message["role"]):
                                        st.write(
                                            message["content"].replace("R$ ", "R\$ ")
                                        )
                                    cont += 1
                            st.session_state.cont_chat = cont
                            st.session_state.atualizar_chat = False

            # st.session_state.messages.clear()
            # st.session_state.messages = st.session_state.messages_backup

            # for message in st.session_state.messages_backup:
            #     with st.chat_message(message["role"]):
            #         st.write(message["content"].replace("R$ ", "R\$ "))

    with tab3:
        st.write("Ainda não implementado.")

    with tab2:
        st.write("Ainda não implementado.")

    if prompt := st.chat_input("🤖: O que você deseja consultar?", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # with st.chat_message("user"):
        #     st.write(prompt)

        response = response_generation(prompt, openai_api_key)

        st.session_state.messages.append({"role": "assistant", "content": response})

        st.toast(response, icon="🤖")

        # with st.chat_message("assistant"):
        #     st.write(response)


if __name__ == "__main__":
    run()


def visualizacoes(df: pd.DataFrame):
    min_year = df["Data_datetime"].dt.year.min()
    max_year = df["Data_datetime"].dt.year.max()

    # Mapeando os números dos meses para abreviações
    month_names = [
        "Jan",
        "Fev",
        "Mar",
        "Abr",
        "Mai",
        "Jun",
        "Jul",
        "Ago",
        "Set",
        "Out",
        "Nov",
        "Dez",
    ]

    col1 = st.columns([1, 1, 1, 1, 1])
    with col1[1]:
        tipo_visualizacao = st.radio("Visualizar por:", ["Contagem", "Valor"])
    with col1[3]:
        tipo_periodo = st.radio(
            "Período:",
            ["Mês (Acumulado)", "Mês", "Bimestre", "Trimestre", "Quadrimestre", "Ano"],
        )

    if tipo_visualizacao == "Contagem":
        agg_func = "count"
        currency_symbol = ""
    else:
        agg_func = "sum"
        currency_symbol = "R$"

    if tipo_periodo == "Mês (Acumulado)":
        mkd_text(
            "Valor Empenhado por Mês (Acumulado)", level="subheader", position="center"
        )
        chart_bar_empenho_periodo(
            df,
            "mes (acumulado)",
            min_year,
            max_year,
            currency_symbol=currency_symbol,
            month_names=month_names,
            agg_func=agg_func,
        )
    elif tipo_periodo == "Mês":
        mkd_text("Valor Empenhado por Mês", level="subheader", position="center")
        chart_bar_empenho_periodo(
            df,
            "mes",
            min_year,
            max_year,
            currency_symbol=currency_symbol,
            month_names=month_names,
            agg_func=agg_func,
        )
    elif tipo_periodo == "Ano":
        mkd_text("Valor Empenhado por Ano", level="subheader", position="center")
        chart_bar_empenho_periodo(
            df,
            "ano",
            min_year,
            max_year,
            currency_symbol=currency_symbol,
            agg_func=agg_func,
        )
    elif tipo_periodo == "Quadrimestre":
        mkd_text(
            "Valor Empenhado por Quadrimestre", level="subheader", position="center"
        )
        chart_bar_empenho_periodo(
            df,
            "quadrimestre",
            min_year,
            max_year,
            currency_symbol=currency_symbol,
            agg_func=agg_func,
        )
    elif tipo_periodo == "Trimestre":
        mkd_text("Valor Empenhado por Trimestre", level="subheader", position="center")
        chart_bar_empenho_periodo(
            df,
            "trimestre",
            min_year,
            max_year,
            currency_symbol=currency_symbol,
            agg_func=agg_func,
        )
    elif tipo_periodo == "Bimestre":
        mkd_text("Valor Empenhado por Bimestre", level="subheader", position="center")
        chart_bar_empenho_periodo(
            df,
            "bimestre",
            min_year,
            max_year,
            currency_symbol=currency_symbol,
            agg_func=agg_func,
        )

    # # Plotagem por Mês - Valor Empenhado
    # mkd_text("Valor Empenhado por Mês", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'mes', min_year, max_year, currency_symbol='R$', month_names=month_names, agg_func='sum')

    # # Plotagem por Mês - Quantidade de Empenhos
    # mkd_text("Quantidade de Empenhos por Mês", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'mes', min_year, max_year, currency_symbol='', month_names=month_names, agg_func='count')

    # # Plotagem por Ano - Valor Empenhado
    # mkd_text("Valor Empenhado por Ano", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'ano', min_year, max_year, currency_symbol='R$', agg_func='sum')

    # # Plotagem por Ano - Quantidade de Empenhos (Opcional)
    # mkd_text("Quantidade de Empenhos por Ano", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'ano', min_year, max_year, currency_symbol='', agg_func='count')

    # # Plotagem por Quadrimestre - Valor Empenhado
    # mkd_text("Valor Empenhado por Quadrimestre", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'quadrimestre', min_year, max_year, currency_symbol='R$', agg_func='sum')

    # # Plotagem por Quadrimestre - Quantidade de Empenhos (Opcional)
    # mkd_text("Quantidade de Empenhos por Quadrimestre", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'quadrimestre', min_year, max_year, currency_symbol='', agg_func='count')

    # # Plotagem por Trimestre - Valor Empenhado
    # mkd_text("Valor Empenhado por Trimestre", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'trimestre', min_year, max_year, currency_symbol='R$', agg_func='sum')

    # # Plotagem por Trimestre - Quantidade de Empenhos (Opcional)
    # mkd_text("Quantidade de Empenhos por Trimestre", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'trimestre', min_year, max_year, currency_symbol='', agg_func='count')

    # # Plotagem por Bimestre - Valor Empenhado
    # mkd_text("Valor Empenhado por Bimestre", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'bimestre', min_year, max_year, currency_symbol='R$', agg_func='sum')

    # # Plotagem por Bimestre - Quantidade de Empenhos (Opcional)
    # mkd_text("Quantidade de Empenhos por Bimestre", level='subheader', position='center')
    # chart_bar_empenho_periodo(df, 'bimestre', min_year, max_year, currency_symbol='', agg_func='count')


def chart_bar_empenho_periodo(
    df_filtered: pd.DataFrame,
    periodo: str,
    min_year: int,
    max_year: int,
    currency_symbol: str = "R$",
    month_names: list = None,
    agg_func: str = "sum",
):
    """
    Cria um gráfico de barras mostrando o valor ou a contagem empenhada por período especificado, com rótulos formatados.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        periodo (str): Tipo de período ('mes', 'ano', 'quadrimestre', 'trimestre', 'bimestre').
        min_year (int): Ano mínimo para filtrar os dados.
        max_year (int): Ano máximo para filtrar os dados.
        currency_symbol (str, opcional): Símbolo da moeda para formatação. Padrão é 'R$'.
        month_names (list, opcional): Lista com nomes abreviados dos meses (necessário para 'mes').
        agg_func (str, opcional): Tipo de agregação ('sum' ou 'count'). Padrão é 'sum'.
    """
    # Verifica se a coluna 'Data_datetime' está no formato datetime
    if not pd.api.types.is_datetime64_any_dtype(df_filtered["Data_datetime"]):
        df_filtered["Data_datetime"] = pd.to_datetime(df_filtered["Data_datetime"])

    # Filtra os dados pelo intervalo de anos
    # df_filtered = df[(df['Data_datetime'].dt.year >= min_year) & (df['Data_datetime'].dt.year <= max_year)].copy()

    # Define o período baseado no tipo especificado
    if periodo.lower() == "mes (acumulado)":
        if month_names is None:
            # Se não forem fornecidos nomes de meses, utiliza os nomes abreviados em português
            month_names = [
                "Jan",
                "Fev",
                "Mar",
                "Abr",
                "Mai",
                "Jun",
                "Jul",
                "Ago",
                "Set",
                "Out",
                "Nov",
                "Dez",
            ]
        df_filtered["Periodo"] = df_filtered["Data_datetime"].dt.month.apply(
            lambda x: month_names[x - 1]
        )
        sort_order = month_names
        label_x = "Mês (Acumulado)"
        titulo = f'{"Quantidade de Empenhos" if agg_func == "count" else "Valor Empenhado"} por Mês ({min_year} ~ {max_year})'
        y_label = "Quantidade" if agg_func == "count" else "Valor Empenhado"
    elif periodo.lower() == "ano":
        df_filtered["Periodo"] = df_filtered["Data_datetime"].dt.year.astype(str)
        sort_order = sorted(df_filtered["Periodo"].unique())
        label_x = "Ano"
        titulo = f'{"Quantidade de Empenhos" if agg_func == "count" else "Valor Empenhado"} por Ano ({min_year} ~ {max_year})'
        y_label = "Quantidade" if agg_func == "count" else "Valor Empenhado"
    elif periodo.lower() == "quadrimestre":
        df_filtered["Quadrimester"] = (
            df_filtered["Data_datetime"].dt.month - 1
        ) // 4 + 1
        df_filtered["Periodo"] = (
            df_filtered["Data_datetime"].dt.year.astype(str)
            + "-Q"
            + df_filtered["Quadrimester"].astype(str)
        )
        sort_order = sorted(df_filtered["Periodo"].unique())
        label_x = "Quadrimestre"
        titulo = f'{"Quantidade de Empenhos" if agg_func == "count" else "Valor Empenhado"} por Quadrimestre ({min_year} ~ {max_year})'
        y_label = "Quantidade" if agg_func == "count" else "Valor Empenhado"
    elif periodo.lower() == "trimestre":
        df_filtered["Trimester"] = (df_filtered["Data_datetime"].dt.month - 1) // 3 + 1
        df_filtered["Periodo"] = (
            df_filtered["Data_datetime"].dt.year.astype(str)
            + "-T"
            + df_filtered["Trimester"].astype(str)
        )
        sort_order = sorted(df_filtered["Periodo"].unique())
        label_x = "Trimestre"
        titulo = f'{"Quantidade de Empenhos" if agg_func == "count" else "Valor Empenhado"} por Trimestre ({min_year} ~ {max_year})'
        y_label = "Quantidade" if agg_func == "count" else "Valor Empenhado"
    elif periodo.lower() == "bimestre":
        df_filtered["Bimester"] = (df_filtered["Data_datetime"].dt.month - 1) // 2 + 1
        df_filtered["Periodo"] = (
            df_filtered["Data_datetime"].dt.year.astype(str)
            + "-B"
            + df_filtered["Bimester"].astype(str)
        )
        sort_order = sorted(df_filtered["Periodo"].unique())
        label_x = "Bimestre"
        titulo = f'{"Quantidade de Empenhos" if agg_func == "count" else "Valor Empenhado"} por Bimestre ({min_year} ~ {max_year})'
        y_label = "Quantidade" if agg_func == "count" else "Valor Empenhado"
    elif periodo.lower() == "mes":
        df_filtered["Mês"] = (df_filtered["Data_datetime"].dt.month - 1) // 1 + 1
        df_filtered["Periodo"] = (
            df_filtered["Data_datetime"].dt.year.astype(str)
            + "-M"
            + df_filtered["Mês"].apply(lambda x: f"{x:02d}")
        )
        sort_order = sorted(df_filtered["Periodo"].unique())
        label_x = "Mês"
        titulo = f'{"Quantidade de Empenhos" if agg_func == "count" else "Valor Empenhado"} por Mês ({min_year} ~ {max_year})'
        y_label = "Quantidade" if agg_func == "count" else "Valor Empenhado"
    else:
        st.error(
            "Tipo de período inválido. Escolha entre 'mes', 'ano', 'quadrimestre', 'trimestre' ou 'bimestre'."
        )
        return

    # Agrupa por 'Periodo' e aplica a agregação especificada
    if agg_func == "sum":
        total_empenhado = (
            df_filtered.groupby("Periodo")["Empenhado_float"].sum().reset_index()
        )
        total_empenhado.rename(
            columns={"Empenhado_float": "Valor_Empenhado"}, inplace=True
        )
    elif agg_func == "count":
        total_empenhado = (
            df_filtered.groupby("Periodo").size().reset_index(name="Quantidade")
        )
    else:
        st.error("Tipo de agregação inválido. Use 'sum' ou 'count'.")
        return

    # Ordena os períodos cronologicamente
    if periodo.lower() == "mes (acumulado)":
        # Ordena pelo índice dos nomes dos meses
        total_empenhado["Ordenacao"] = total_empenhado["Periodo"].apply(
            lambda x: sort_order.index(x)
        )
    elif periodo.lower() in ["ano", "quadrimestre", "trimestre", "bimestre", "mes"]:
        # Ordena alfanumericamente
        total_empenhado["Ordenacao"] = total_empenhado["Periodo"].apply(
            lambda x: sort_order.index(x)
        )
    total_empenhado = total_empenhado.sort_values("Ordenacao")
    total_empenhado = total_empenhado.drop(columns=["Ordenacao"])

    # Aplica a formatação de moeda para os textos, se for soma
    if agg_func == "sum":
        total_empenhado["Valor Empenhado"] = total_empenhado["Valor_Empenhado"].apply(
            lambda x: format_currency(x, currency_symbol)
        )
    elif agg_func == "count":
        total_empenhado["Quantidade de Empenhos"] = total_empenhado[
            "Quantidade"
        ].astype(str)

    # Define os dados para plotagem
    if agg_func == "sum":
        x_data = "Periodo"
        y_data = "Valor_Empenhado"
        text_data = "Valor Empenhado"
    elif agg_func == "count":
        x_data = "Periodo"
        y_data = "Quantidade"
        text_data = "Quantidade de Empenhos"

    # Cria o gráfico de barras
    fig = px.bar(
        total_empenhado,
        x=x_data,
        y=y_data,
        labels={"Periodo": label_x, y_data: y_label},
        text=text_data,
        category_orders={"Periodo": sort_order},
    )

    # Atualiza o layout para melhor visualização
    fig.update_layout(
        xaxis=dict(type="category"),
        xaxis_title=label_x,
        yaxis_title=y_label,
        title=titulo,
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    # Ajusta a posição do texto e a formatação
    fig.update_traces(
        textposition="outside",
        texttemplate="%{text}",
        textfont=dict(color="#000000"),
        marker_color="#0000ff",  # Opcional: Define a cor das barras, poderia ser uma cor, ex: 'blue'
    )

    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # # Opcional: Adiciona um botão para baixar os dados agrupados como CSV
    # csv = total_empenhado.to_csv(index=False).encode('utf-8')
    # st.download_button(
    #     label="Baixar Dados como CSV",
    #     data=csv,
    #     file_name=f'empenhos_por_{periodo}.csv',
    #     mime='text/csv',
    # )

    # # Opcional: Adiciona um botão para baixar o gráfico como PNG
    # img_bytes = pio.to_image(fig, format='png')
    # st.download_button(
    #     label="Baixar Gráfico como PNG",
    #     data=img_bytes,
    #     file_name=f'empenhos_por_{periodo}.png',
    #     mime='image/png',
    # )
