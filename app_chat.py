import streamlit as st
import numpy as np  
import random
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()



def response_generation():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def response_generation_openai(prompt:str, client:OpenAI, model:str="gpt-3.5-turbo"):
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model= st.session_state["openai_model"],
            messages = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
    


def run():
    st.title("ChatGPT-like clone")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-3.5-turbo"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt:= st.chat_input("Whats is up?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response_generation_openai(prompt, client)


def run_with_simple_chat_streaming():

    st.title("Simple chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response = st.write_stream(response_generation())
        
        st.session_state.messages.append({"role": "assistant", "content": response})



def run_bot_interacao_simples_nao_pensa():
    st.title("Echo Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    
    
    

def run2():
    st.title("Chat bases")
    
    st.divider()

    with st.chat_message("user"):
        st.write("Olá, tudo bem?")
        
    message = st.chat_message("assistant")
    message.write("Hello human")
    message.bar_chart(np.random.randn(30, 3))
    
    prompt = st.chat_input("Diga Alguma coisa")
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")

if __name__ == '__main__':
    run()