import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st

# Agent and LLM
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ConversationalAgent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
# Memory
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
# Tools
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import OpenWeatherMapAPIWrapper

import os
import requests

from dotenv import load_dotenv



load_dotenv()

secret = os.getenv("API_SECRET")
# Set Tools
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=OPENWEATHERMAP_API_KEY)

openai_api_key = os.getenv("OPENAI_API_KEY")

def consultar_subelementos(query: str) -> str:
    import requests

    url = "https://api.controlgov.org/embeddings/subelementos"
    payload = {
        "query": query,
        "secret": secret  # É recomendável armazenar o secret em uma variável de ambiente
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers, timeout=10)

    if response.status_code == 200:
        data = response.json()
        return data.get("resposta", "Nenhuma resposta encontrada.")
    else:
        return f"Erro ao consultar a API: {response.status_code}"

def consultar_empenhado_sum(query=None):
    url = 'https://api.controlgov.org/elementos/despesa/empenhado-sum/'
    headers = {
        'accept': 'application/json'
    }
    
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
    url = 'https://api.controlgov.org/elementos/despesa/empenhado-sum/'
    headers = {
        'accept': 'application/json'
    }
    
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
                elem for elem in elementos 
                if query.lower() in elem.get("elemento_de_despesa", "").lower()
            ]
            if not elementos:
                return f"Nenhum empenho encontrado para o elemento de despesa: '{query}'."
        
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

def identificar_categoria(query: str) -> str:
    elementos_keywords = ["elemento", "elemento de despesa", "elemento de"]
    subelementos_keywords = ["subelemento", "sub", "sub elemento", "sub-elemento"]

    # Converter a consulta para minúsculas para facilitar a comparação
    query_lower = query.lower()

    # Verificar presença de palavras-chave de elementos
    if any(keyword in query_lower for keyword in elementos_keywords):
        return "Consulta Tipo 2: Consulte por subelemento"

    # Verificar presença de palavras-chave de subelementos
    if any(keyword in query_lower for keyword in subelementos_keywords):
        return "Consulta Tipo 1: Consulte por elemento"

    # Se não identificar, solicitar mais informações
    return "necessita_informacao. Pergunte se é um elemento ou subelemento."

def identificar_categoria_tool(query: str) -> str:
    categoria = identificar_categoria(query)
    if categoria == "elemento":
        return "A consulta refere-se a um **elemento** de despesa."
    elif categoria == "subelemento":
        return "A consulta refere-se a um **subelemento** de despesa."
    else:
        return "ERRO: Necessita de mais informações para identificar a categoria da consulta.\n Não consegui identificar se a consulta refere-se a um elemento ou subelemento de despesa."





def generate_response_agent(text):
    
    # Definir as ferramentas
    subelementos_tool = Tool(
        name="Consultar Subelemento Individualmente",
        func=consultar_subelementos,
        description=(
            "Use esta ferramenta para obter informações sobre alguns subelementos financeiros. "
            "Por exemplo, você pode perguntar: 'Qual o total empenhado para o subelmento <subelemento>?'"
        )
    )

    empenhos_por_elemento_tool = Tool(
        name="Consultar Todos os Elementos de uma Vez",
        func=listar_empenhos_por_elemento,
        description=(
            "Use esta ferramenta para obter a lista de empenhos por elemento de despesa. "
            "Por exemplo, você pode perguntar: 'Quais são os empenhos para o elemento de despesa por Obrigação Patronal?' "
        )
    )
            # "Ou simplesmente: 'Liste todos os empenhos por elemento de despesa.'"
    
    categoria_tool = Tool(
        name="Identificar Categoria",
        func=identificar_categoria_tool,
        description=(
            "Use esta ferramenta para identificar se a consulta do usuário refere-se a um elemento ou subelemento de despesa. "
            "Se não for possível identificar, informe que são necessárias mais informações."
        )
    )

    tools = [
        subelementos_tool,
        # empenhado_sum_tool,
        empenhos_por_elemento_tool,
        categoria_tool  # Adiciona a nova ferramenta aqui
    ]

    prefix = """Você é um assistente amigável especializado em finanças governamentais.
    Você pode ajudar os usuários a consultar informações sobre elementos e subelementos de despesa.
    Você tem acesso às seguintes ferramentas:
    
    Sempre comece identificando a categoria da consulta (elemento ou subelemento).
    - Identificar Categoria: Use esta ferramenta para identificar se a consulta refere-se a um elemento ou subelemento de despesa. Se não for possível identificar, informe que são necessárias mais informações.
    
    - Consulta Tipo 1:
        - Consultar Subelemento Individualmente: Use esta ferramenta para obter informações sobre valores empenhados por subelementos de despesa.

    - Consulta Tipo 2:
        - Consultar Todos os Elementos de uma Vez: Use esta ferramenta para obter a lista de valores empenhados por elemento de despesa.
    """
        # - Consultar Empenhado Sum: Use esta ferramenta para obter a soma de todos os valores empenhados para cada elemento de despesa.

    suffix = """
    Histórico do Chat:
    {chat_history}
    Última Pergunta: {input}
    {agent_scratchpad}
    Sempre responda em Português.
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
            messages=msg,
            memory_key="chat_history",
            return_messages=True
        )
    memory = st.session_state.memory
    

    # Configurar o LLM
    llm_chain = LLMChain(
        llm=ChatOpenAI(temperature=0.8, model_name="gpt-4o-mini"),
        prompt=prompt,
        verbose=True
    )

    # Configurar o agente
    agent = ConversationalAgent(
        llm_chain=llm_chain,
        memory=memory,
        verbose=True,
        max_interactions=3,
        tools=tools
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

    # Executar o agente
    if text:
        result = agent_executor.run(text)
        
        # Verificar se a resposta é a solicitação de mais informações
        if "não consegui identificar" in result.lower():
            return result  # Retorna imediatamente a mensagem solicitando mais informações
        else:
            return result

    
    # with st.expander("My thinking"):
    #     st.write(st.session_state.memory.chat_memory.messages)

# def response_generation_openais(prompt:str, client:OpenAI, model:str="gpt-3.5-turbo"):
#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model= st.session_state["openai_model"],
#             messages = [
#                 {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
#             ],
#             stream=True,
#         )
#         response = st.write_stream(stream)
#     st.session_state.messages.append({"role": "assistant", "content": response})
    

def response_generation(text: str, openai_api_key):
    with st.spinner("Estou pensando..."):
        response = generate_response_agent(text)
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

    
def run(openai_api_key):
    st.title("🦜🔗 Quickstart App")
    
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-3.5-turbo"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"].replace("R$ ", "R\$ "))
    
    if prompt := st.chat_input("O que você deseja consultar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        response_generation(prompt, openai_api_key)

            

if __name__ == "__main__":
    run(openai_api_key)
    # if st.session_state.memory:
    #     st.write(st.session_state.memory.chat_memory.messages)
    # # Deixa apenas as 2 últimas mensagens encaminhadas pelo HumanMessage
    
    # save_messages = st.session_state.memory.chat_memory.messages[-4:]
    # for msg in save_messages:
    #     if msg.type!="human":
    #         save_messages.remove(msg)
    # st.write(save_messages)
    # st.session_state.memory.chat_memory.messages = save_messages
            
            
        
    # for msg in save_messages:
    #     if msg["role"] == "user":
    #         st.session_state.messages.append(msg)
    # # Salva as mensagens no estado da sessão
    # st.session_state.memory.chat_memory.messages = save_messages
    