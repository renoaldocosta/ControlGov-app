import streamlit as st
from app.services.text_functions import mkd_text_divider, mkd_text, mkd_paragraph
import pandas as pd
import pandas as pd
import os
from dotenv import load_dotenv
from pymongo import MongoClient

def testa_conexao_mongodb(db:str, collection:str):
    # Carregar automaticamente o arquivo .env no mesmo diretório ou em diretórios pais
    load_dotenv()

    # pega db_password do ambiente
    db_password = os.environ.get('db_password')

    uri = f"mongodb+srv://renoaldo_teste:{db_password}@cluster0.zmdkz1p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    # Create a new client and connect to the server
    client = MongoClient(uri)

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        db = client[db]
        collection = db[collection]
        return collection
    except Exception as e:
        print(e)
        raise SystemExit("Unable to connect to the database. Please check your URI.")

@st.cache_data
def get_empenhos(db, collection):
    mongodb_collection = testa_conexao_mongodb(db, collection)
    if 'df_empenhos' not in st.session_state:
        df_empenhos = pd.DataFrame(list(mongodb_collection.find()))
    
        # Tratar a coluna 'Item(ns)' para evitar mistura de tipos
        if 'Item(ns)' in df_empenhos.columns:
        
            df_empenhos['Item(ns)'] = df_empenhos['Item(ns)'].apply(lambda x: str(x) if isinstance(x, list) else x)

        st.session_state.df_empenhos = df_empenhos
    else:
        df_empenhos = st.session_state.df_empenhos
        return df_empenhos

def run():
    mkd_text("Câmara Municipal de Pinhão - SE", level='title', position='center')
    # Define database and collection
    db, collection = 'CMP', 'EMPENHOS_DETALHADOS_STAGE'
    # Test connection to MongoDB
    mongodb_collection = testa_conexao_mongodb(db, collection)
    # Retrieve data from MongoDB as a DataFrame
    df_empenhos = get_empenhos(db, collection)
    
    with st.expander("Amostra dos dados: Empenhos"):
        st.dataframe(df_empenhos.sample(5).reset_index(drop=True))
    
    mkd_text_divider("Relatório", level='header', position='center')

if __name__ == "__main__":
    run()