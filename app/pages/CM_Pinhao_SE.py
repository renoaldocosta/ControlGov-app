import streamlit as st
from app.services.text_functions import mkd_text_divider, mkd_text, mkd_paragraph
import pandas as pd
import pandas as pd
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pygwalker.api.streamlit import StreamlitRenderer

# Dicionário para traduzir os meses para português
month_translation = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
}

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



def format_df(df_empenhos):
    df = df_empenhos.copy()

    df = df.drop(columns=['_id'])
    
    df['Credor'] = df['Credor'].str.split(' - ').apply(lambda x: f'{x[1]} - {x[0]}' if len(x) > 1 else x[0])

    # Remover o símbolo de moeda 'R$' e os pontos dos milhares
    value_columns = ['Alteração', 'Empenhado', 'Liquidado', 'Pago' ]
    for colunm in value_columns:
        df[colunm] = df[colunm].str.replace(r'R\$ ?', '', regex=True).str.replace('.', '').str.replace(',', '.')
        df[colunm] = df[colunm].astype(float)

    data_columns = ['Data', 'Atualizado'] 
    for column in data_columns:
        df[column] = pd.to_datetime(df[column])
        
    return df


@st.cache_data
def get_empenhos(db, collection):
    mongodb_collection = testa_conexao_mongodb(db, collection)
    if 'df_empenhos' not in st.session_state:
        df_empenhos = pd.DataFrame(list(mongodb_collection.find()))
    
        # Tratar a coluna 'Item(ns)' para evitar mistura de tipos
        if 'Item(ns)' in df_empenhos.columns:
        
            df_empenhos['Item(ns)'] = df_empenhos['Item(ns)'].apply(lambda x: str(x) if isinstance(x, list) else x)
            
        df_empenhos = format_df(df_empenhos)
                
        return df_empenhos
    else:
        df_empenhos = st.session_state.df_empenhos
        return df_empenhos


def metrics(df):
    mkd_text_divider("Métricas", level='header', position='center')
    
    first_year = df['Data'].dt.year.min()
    last_year = df['Data'].dt.year.max()
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.metric("Total de Empenhos", df.shape[0])
        
    with col2:
        st.metric("Período", f'{first_year}~{last_year}')
        
    col1, col2, col3, col4 = st.columns(4)
    


def year_filter(df):
    # Garantir que a coluna 'Data' está no formato datetime
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    # Definir o intervalo de anos para o slider
    first_year = df['Data'].dt.year.min()
    last_year = df['Data'].dt.year.max()

    # Slider para selecionar o intervalo de anos
    selected_years = st.slider("Ano", min_value=first_year, max_value=last_year, value=(first_year, last_year))

    # Filtrar o DataFrame pelo intervalo de anos
    df = df[df['Data'].dt.year.between(*selected_years)]
    
    st.session_state['selected_years'] = selected_years
        
    return df

def month_filter(df):
    # Adicionar coluna com o número do mês para ordenação
    df['Mês_Numero'] = df['Data'].dt.month
    
    # Traduzir os meses para português
    df['Mês'] = df['Mês_Numero'].map(month_translation)

    # Lista de meses em ordem correta
    all_months = list(month_translation.values())

    # Filtro por mês, com os meses em ordem correta
    months = st.multiselect("Mês", all_months, placeholder="Selecione um ou mais meses")

    # Aplicar o filtro por mês
    if months:
            df = df[df['Mês'].isin(months)]
    
    # Ordenar o DataFrame pelo número do mês (Mês_Numero)
    df = df.sort_values(by='Mês_Numero')
    
    # Salvar os meses selecionados no session_state
    st.session_state['selected_months'] = months
    
    return df


def credores(df):
    credores = list(df['Credor'].unique())
    
    selected_creditors = st.multiselect("Credores / Fornecedores", credores, placeholder="Selecione um ou mais credores")
    
    if selected_creditors:
            df = df[df['Credor'].isin(selected_creditors)]
    
    # Atualiza o session_state
    st.session_state['credores'] = df
    
    return df


def elemento_despesa(df):
    elementos = list(df['Elemento de Despesa'].unique())
    
    selected_elementos = st.multiselect("Elemento de Despesa", elementos, placeholder="Selecione um ou mais elementos de despesa")
    
    if selected_elementos:
        df = df[df['Elemento de Despesa'].isin(selected_elementos)]    
    # Atualiza o session_state
    st.session_state['elemento_despesa'] = df
    
    return df


def sub_elemento_despesa(df):
    pass
    sub_elementos = list(df['Subelemento'].unique())
    
    selected_sub_elementos = st.multiselect("Subelemento de Despesa", sub_elementos, placeholder="Selecione um ou mais subelementos de despesa")
    
    if selected_sub_elementos:
        df = df[df['Subelemento'].isin(selected_sub_elementos)]    
    # Atualiza o session_state
    st.session_state['sub_elemento_despesa'] = df
    
    return df


def filters():
    with st.sidebar:
        try:
            df = st.session_state.df_empenhos
            
            st.markdown("## Filtros")
            
            df = year_filter(df)
            df = month_filter(df)
            df = credores(df)
            df = elemento_despesa(df)
            df = sub_elemento_despesa(df)
            
            return df
        except Exception as e:
            
            pass
            st.write(f"Erro ao carregar os dados. {e}")
            


def pygwalker(df):
    #from pygwalker.api.streamlit import StreamlitRenderer
    if not df.empty:
        # Converter colunas de Timestamp para string
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        pyg_app = StreamlitRenderer(df, appearance='light',theme_key='vega')
        pyg_app.explorer()


def run():
    
        
    
    mkd_text("Câmara Municipal de Pinhão - SE", level='title', position='center')
    # Define database and collection
    db, collection = 'CMP', 'EMPENHOS_DETALHADOS_STAGE'
    # Test connection to MongoDB
    mongodb_collection = testa_conexao_mongodb(db, collection)
    # Retrieve data from MongoDB as a DataFrame
    df_empenhos = get_empenhos(db, collection)
    st.session_state['df_empenhos'] = df_empenhos
    
    tab1, tab2 = st.tabs(['Empenhos', 'Exploração'])
    
    df = filters()
    
    with tab1:
        st.write(df.head())
    with tab2:
        pygwalker(df_empenhos)

    with st.expander("Amostra dos dados: Empenhos"):
        st.dataframe(df.head().reset_index(drop=True))
    

    metrics(df)
    
if __name__ == "__main__":
    run()