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


def generate_response_agent(text):
    
    # Set Tools
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

    search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=OPENWEATHERMAP_API_KEY)
    # Definir a nova Tool
    subelementos_tool = Tool(
        name="Consultar Subelementos",
        func=consultar_subelementos,
        description=( "Use this tool to obtain information about financial sub-elements."
                    "For example, you can ask: 'Qual o total empenhado para Fretes?'" 
        )
    )
    
    # Definir a ferramenta para consultar empenhado sum
    empenhado_sum_tool = Tool(
        name="Consultar Empenhado Sum",
        func=consultar_empenhado_sum,
        description=(
            "Use esta ferramenta para obter a soma de todos os valores empenhados para cada "
            "elemento de despesa. Por exemplo, você pode perguntar: 'Qual a soma empenhada por elemento de despesa?'"
        )
    )
    
    # Definir a nova ferramenta para consultar empenhos por elemento
    empenhos_por_elemento_tool = Tool(
        name="Consultar Empenhos por Elemento",
        func=listar_empenhos_por_elemento,
        description=(
            "Use esta ferramenta para obter a lista de empenhos por elemento de despesa. "
            "Por exemplo, você pode perguntar: 'Quais são os empenhos para o elemento de despesa Fretes?' "
            "Ou simplesmente: 'Liste todos os empenhos por elemento de despesa.'"
        )
    )



    tools = [
        # Tool(
        #     name="Search",
        #     func=search.run,
        #     description="Useful for when you need to get current, up to date answers.",
        # ),
        # Tool(
        #     name="Weather",
        #     func=weather.run,
        #     description="Useful for when you need to get the current weather in a location.",
        # ),
        subelementos_tool,
        empenhado_sum_tool,
        empenhos_por_elemento_tool
    ]

    # Set Chat Conversation
    # - Search: Useful for when you need to get current, up to date answers.
    # - Weather: Useful for when you need to get the current weather in a location.
    prefix = """You are a friendly modern day planner.
    You can help users to find activities in a given city based
    on their preferences and the weather.
    You have access to the following tools:

    - Consultar Subelementos: Use esta ferramenta para obter informações sobre subelementos financeiros.
    - Consultar Empenhado Sum: Use esta ferramenta para obter a soma de todos os valores empenhados para cada elemento de despesa.
    - Consultar Empenhos por Elemento: Use esta ferramenta para obter a lista de empenhos por elemento de despesa.
    """

    suffix = """
    Chat History:
    {chat_history}
    Latest Question: {input}
    {agent_scratchpad}
    Always respond in Portuguese.
    """

    prompt = ConversationalAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input",
                        "chat_history",
                        "agent_scratchpad"],
    )

    # Set Memory

    msg = StreamlitChatMessageHistory()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            messages=msg,
            memory_key="chat_history",
            return_messages=True
        )
    memory = st.session_state.memory

    # Set Agent

    llm_chain = LLMChain(
        llm=ChatOpenAI(temperature=0.8, model_name="gpt-4o-mini"),
        prompt=prompt,
        verbose=True
    )

    agent = ConversationalAgent(
        llm_chain=llm_chain,
        memory=memory,
        verbose=True,
        max_interactions=3,
        tools=tools
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                        tools=tools,
                                                        memory=memory,
                                                        verbose=True)

    # query = st.text_input("O que você quer fazer hoje?", placeholder="Digite aqui...")

    # if text:
    #     with st.spinner("Estou pensando..."):
    #         result = agent_executor.run(text)
    #         st.info(result, icon="🤖")

    if text:
        result = agent_executor.run(text)
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
    

def response_generation(text:str, openai_api_key):
    model = ChatOpenAI(
        api_key = openai_api_key,
        model = "gpt-3.5-turbo",
        temperature=0.7,
    )
    messages = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ],
    prompt = ChatPromptTemplate.from_template("Responda como um pirata: {message}. Portugues Brasil.")
    parser = StrOutputParser()
    chain = prompt | model | parser  # Define the chain
    # stream = chain.stream({"message": f"{text}"})  # Stream output


    # with st.spinner("Pensando..."):
        # response = model.invoke(text).content
def response_generation(text:str, openai_api_key):
    with st.spinner("Estou pensando..."):
        response = generate_response_agent(text)
    
    
    with st.chat_message("assistant"):
        st.write(response)
    
    # st.info(response)
    # responses = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

    # # Função geradora para dividir os chunks em palavras
    # def split_into_words(stream):
    #     buffer = ""
    #     for chunk in stream:
    #         buffer += chunk
    #         while ' ' in buffer:
    #             word, buffer = buffer.split(' ', 1)
    #             yield word + ' '
    #     if buffer:
    #         yield buffer

    # # Usar a função geradora no write_stream
    # responses = st.write_stream((stream))
    # st.session_state.messages.append({"role": "assistant", "content": responses})
    
    
def run(openai_api_key):
    st.title("🦜🔗 Quickstart App")
    
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-3.5-turbo"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt:= st.chat_input("Whats is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        response_generation(prompt, openai_api_key)
    
    # with st.form("my_form"):
    #     text = st.text_area(
    #         "Enter text:",
    #         "Quais são os três conselhos principais para quem quer aprender langchain?"
    #     )
        
    #     submitted = st.form_submit_button("Submit")
        
        
    #     if not openai_api_key.startswith("sk-"):
    #         st.warning("Please enter your OpenAI API key!", icon="🔑")
        
    #     if submitted and openai_api_key.startswith("sk-"):
    #         with st.spinner("Generating response..."):
    #             response_generation(text, openai_api_key)
            

if __name__ == "__main__":
    run(openai_api_key)