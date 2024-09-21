import streamlit as st
from app.services.text_functions import mkd_text_divider, mkd_text, mkd_paragraph
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

def run():
    # Conectar ao MongoDB e buscar os dados
    mongodb_collection = testa_conexao_mongodb('CMP', 'EMPENHOS_DETALHADOS_STAGE')
    df_empenhos = pd.DataFrame(list(mongodb_collection.find()))

    # Tratar colunas com listas (se houver), transformando-as em strings
    for col in df_empenhos.columns:
        if df_empenhos[col].apply(lambda x: isinstance(x, list)).any():
            df_empenhos[col] = df_empenhos[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

    # Exibir o DataFrame tratado
    st.dataframe(df_empenhos)

if __name__ == '__main__':
    run()
