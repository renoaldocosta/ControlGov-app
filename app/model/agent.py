import os
import requests

from dotenv import load_dotenv

import streamlit as st

# langchain
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

# LangChain Community
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# LangChain OpenAI
from langchain_openai.chat_models import ChatOpenAI


# Carregar variáveis de ambiente
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
secret = os.getenv("SECRET")

# Função que consulta a API do ControlGov para obter informações sobre CPF ou CNPJ
def consultar_cpf_cnpj(query: str) -> str:
    import requests

    url = "https://api.controlgov.org/embeddings/subelementos"
    payload = {"query": query, "secret": secret}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        return data.get("resposta", "Nenhuma resposta encontrada.")
    else:
        return f"Erro ao consultar a API: {response.status_code}"

# Função que consulta a API do ControlGov para obter informações sobre pessoas físicas e jurídicas
def consultar_PessoaFisica_PessoaJuridica(query: str) -> str:
    import requests

    url = "https://api.controlgov.org/embeddings/subelementos"
    payload = {"query": query, "secret": secret}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        return data.get("resposta", "Nenhuma resposta encontrada.")
    else:
        return f"Erro ao consultar a API: {response.status_code}"

# Função que consulta a API do ControlGov para obter informações sobre subelementos financeiros
def consultar_subelementos(query: str) -> str:
    import requests

    url = "https://api.controlgov.org/embeddings/subelementos"
    payload = {"query": query, "secret": secret}  # É recomendável armazenar o secret em uma variável de ambiente
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        data["resposta"] = data["resposta"] + "\nFormatar valores em R$"
        return data.get("resposta", "Nenhuma resposta encontrada.")
    else:
        return f"Erro ao consultar a API: {response.status_code}"

# Função que consulta a API do ControlGov para obter a soma dos valores empenhados
def consultar_empenhado_sum(query=None):
    import requests

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
            resultado += f"• {elemento_despesa}: R$ {total_empenhado:,.2f}\n"

        return resultado

    except requests.exceptions.RequestException as e:
        return f"Ocorreu um erro ao consultar a API: {e}"
    except ValueError:
        return "Erro ao processar a resposta da API."

# Função que lista os empenhos por elemento de despesa
def listar_empenhos_por_elemento(query=None):

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
                return f"Nenhum empenho encontrado para o elemento de despesa: '{query}'."

        resultado = "Empenhos por Elemento de Despesa:\n\n"
        for elemento in elementos:
            elemento_despesa = elemento.get("elemento_de_despesa", "Desconhecido")
            total_empenhado = elemento.get("total_empenhado", 0)
            resultado += f"• {elemento_despesa}: R$ {total_empenhado:,.2f}\n"

        return resultado

    except requests.exceptions.RequestException as e:
        return f"Ocorreu um erro ao consultar a API: {e}"
    except ValueError:
        return "Erro ao processar a resposta da API."

# Função que gera a resposta do agente de atendimento
def load_agent(text):
    # Consultar CPF ou CNPJ
    text = "Me responda apenas:\n" + text

    consultar_cpf_cnpj_tool = Tool(
        name="Consultar CPF ou CNPJ",
        func=consultar_cpf_cnpj,
        description=(
            "Use esta ferramenta para obter informações sobre CPF ou CNPJ de um credor."
            "Por exemplo, você pode perguntar: 'Qual o CPF do credor <nome> com asteriscos?' ou 'Qual o CNPJ do credor <nome>?'"
            "Se o usuário não informar o nome do credor, o agente solicitará o nome do credor."
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
        name="Consultar o total empenhado para todos os Elementos de uma Vez",
        func=listar_empenhos_por_elemento,
        description=(
            "Use esta ferramenta para obter um jso com uma lista de empenhos por elemento de despesa. "
        ),
    )

    tools = [
        subelementos_tool,
        # empenhado_sum_tool,
        empenhos_por_elemento_tool,
        empenho_pessoa_fisica_juridica,
        consultar_cpf_cnpj_tool,
        # categoria_tool  # Adiciona a nova ferramenta aqui
    ]


    prefix = """# Assistente de Finanças Governamentais da Câmara Municipal de Pinhão/SE
    Você é um assistente direto e especializado em finanças governamentais.

    Você pode ajudar os usuários a consultar informações da Câmara Municipal de Pinhão/SE sobre:

    - Elementos e subelementos de despesa
    - Consultas aos valores empenhados a Pessoas Físicas e Jurídicas
    - Consultas a CPF ou CNPJ dos credores

    ## Ferramentas Disponíveis

    Você tem acesso às seguintes ferramentas:

    1. Consultar Empenho a Pessoa Física ou Jurídica
    - *Descrição:* Use esta ferramenta para obter informações sobre valores empenhados para PF ou PJ.
    
    2. Consultar CPF ou CNPJ
    - *Descrição:* Use esta ferramenta para obter informações sobre CPF ou CNPJ dos credores.
    
    3. Consultar Subelemento Individualmente
    - *Descrição:* Use esta ferramenta para obter informações sobre valores empenhados por subelementos de despesa.
    
    4. Consultar o total empenhado para todos os Elementos de uma Vez
    - *Descrição:* Use esta ferramenta para obter a lista de valores empenhados por elemento de despesa.

    ## Instruções para Uso das Ferramentas

    Para usar uma ferramenta, responda exatamente neste formato:
    Pensamento: [Seu raciocínio sobre a próxima ação a ser tomada] 
    Ação: [O nome da ferramenta a ser usada] 
    Entrada da Ação: [Os dados de entrada necessários para a ferramenta] 
    Observação: [O resultado da execução da ferramenta]
    Resposta Final: [Sua resposta ao usuário]
    
    
    Se você já tiver todas as informações necessárias para responder à pergunta do usuário, forneça a resposta final:
    Pensamento: Já tenho as informações necessárias para responder ao usuário. 
    """
    suffix = """

    Histórico do Chat:

    {chat_history}

    Última Pergunta: {input}

    {agent_scratchpad}

    Sempre responda em Português.

    Responda apenas ao que foi perguntado. Evite demais informações.
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
        agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True, handle_errors=True
    )
    
    return agent_executor