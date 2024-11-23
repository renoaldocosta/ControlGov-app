import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()


def response_generation(text:str):
    model = ChatOpenAI(
        api_key = st.secrets["OPENAI_API_KEY"],
        model = "gpt-3.5-turbo",
        temperature=0.7,
    )
    response = model.invoke(text).content
    st.info(response)
    

def run():
    st.title("🦜🔗 Quickstart App")
    
    with st.form("my_form"):
        text = st.text_area(
            "Enter text:",
            "Quais são os três conselhos principais para quem quer aprender langchain?"
        )
        
        submitted = st.form_submit_button("Submit")

        if not openai_api_key.startswith("sk-"):
            st.warning("Please enter your OpenAI API key!", icon="🔑")
        
        if submitted and openai_api_key.startswith("sk-"):
            with st.spinner("Generating response..."):
                response_generation(text)
            

if __name__ == "__main__":
    run()