import streamlit as st
from app.services.text_functions import mkd_text_divider, mkd_text, mkd_paragraph
import pandas as pd
import pandas as pd
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit as st
from st_aggrid import AgGrid

import pandas as pd
import locale

# Definir o locale para o Brasil
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

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
    if 'df_escolhido' in st.session_state:
        df_escolhido = st.session_state['df_escolhido']
        mkd_text_divider(f"Métricas de {df_escolhido}", level='subheader', position='center')
    
    
    # Converter colunas de data para o tipo datetime
    df['data'] = pd.to_datetime(df['data'])
    df['dataEmpenho'] = pd.to_datetime(df['dataEmpenho'])

    # Cálculo das métricas
    total_registros = df.shape[0]
    data_mais_recente = df['data'].max().strftime('%d/%m/%Y')  # Converte a data para string
    data_mais_antiga = df['data'].min().strftime('%d/%m/%Y')  # Converte a data para string
    valor_minimo = df['valor'].min()
    valor_medio = df['valor'].mean()
    valor_maximo = df['valor'].max()


    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", total_registros)
    with col2:
        st.metric("Data Mais Antiga", data_mais_antiga)
    with col3:
        st.metric("Data Mais Recente", data_mais_recente)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Valor Mínimo", f"{locale.currency(valor_minimo, grouping=True)}")
    with col5:
        st.metric("Valor Médio", f"{locale.currency(valor_medio, grouping=True)}")
    with col6:
        st.metric("Valor Máximo", f"{locale.currency(valor_maximo, grouping=True)}")
    
    
    
    
    


def year_filter(df):
    # Garantir que a coluna 'Data' está no formato datetime
    df['data'] = pd.to_datetime(df['data'], errors='coerce')

    # Definir o intervalo de anos para o slider
    first_year = df['data'].dt.year.min()
    last_year = df['data'].dt.year.max()
    

    # Filtrar o DataFrame pelo intervalo de anos
    if first_year != last_year:
        # Slider para selecionar o intervalo de anos
        selected_years = st.slider("Ano", min_value=first_year, max_value=last_year, value=(first_year, last_year))
        df = df[df['data'].dt.year.between(*selected_years)]
        
    else:
        st.caption(f'Ano selecionado: {first_year}')
        selected_years = first_year
        
    st.session_state['selected_years'] = selected_years
        
    return df

def month_filter(df):
    # Adicionar coluna com o número do mês para ordenação
    df['Mês_Numero'] = df['data'].dt.month
    
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
    credores = list(df['credor'].unique())
    
    selected_creditors = st.multiselect("Credores / Fornecedores", credores, placeholder="Selecione um ou mais credores")
    
    if selected_creditors:
            df = df[df['credor'].isin(selected_creditors)]
    
    # Atualiza o session_state
    st.session_state['credor'] = df
    
    return df

def orgao(df):
    orgaos = list(df['orgao'].unique())
    
    selected_orgaos = st.multiselect("Órgão", orgaos, placeholder="Selecione um ou mais órgãos")
    
    if selected_orgaos:
            df = df[df['orgao'].isin(selected_orgaos)]
    
    # Atualiza o session_state
    st.session_state['orgao'] = df
    
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
    sub_elementos = sorted(list(df['Subelemento'].unique()))
    
    selected_sub_elementos = st.multiselect("Subelemento de Despesa", sub_elementos, placeholder="Selecione um ou mais subelementos de despesa")
    
    if selected_sub_elementos:
        df = df[df['Subelemento'].isin(selected_sub_elementos)]    
    # Atualiza o session_state
    st.session_state['sub_elemento_despesa'] = df
    
    return df


def Categorias_de_base_legal(df):
    #base_legal = sorted(set(['DISPENSADO'] + list(df[df['Categorias de base legal'].str.split('/').str[0] != 'DISPENSADO']['Categorias de base legal'])))
    base_legal = sorted(set(list(df['Categorias de base legal'])))
    # Exibe o seletor múltiplo com as opções filtradas
    selected_base_legal = st.multiselect("Base Legal", base_legal, placeholder="Selecione uma ou mais bases legais")

    # Se uma ou mais bases legais forem selecionadas, filtra o DataFrame
    if selected_base_legal:
        
        df = df[df['Categorias de base legal'].isin(selected_base_legal)]

    # Atualiza o session_state com o DataFrame filtrado
    st.session_state['base_legal'] = df

    return df


def converter_valor_brasileiro(valor_str):
    # Remove o ponto dos milhares e substitui a vírgula por ponto
    valor_str = valor_str.replace('.', '').replace(',', '.')
    try:
        return float(valor_str)
    except ValueError:
        st.error("O valor inserido não é válido. Verifique o formato.")
        return None

def valores(df):
    allow_filter_value = st.checkbox("Filtrar dados por valor", value=False)
    
    if not allow_filter_value:
        return df
    else:
        # Filtrar por valores
        tipo = st.radio("Tipo de Filtro", ['Menor', 'Igual', 'Maior'], horizontal=True)
        
        # Permitir que o usuário cole o valor com formato brasileiro
        valor_colado = st.text_input("Valor", value="0,00")  # Coloque o valor padrão no formato brasileiro
        if valor_colado == '':
            valor_colado = "0,00"
        
        # Converter o valor colado para um número decimal
        filter_value = converter_valor_brasileiro(valor_colado)
        
        if filter_value is not None:
            if tipo == 'Menor':
                df_filtrado = df[df['valor'] < filter_value]
            elif tipo == 'Igual':
                df_filtrado = df[df['valor'] == filter_value]
            else:
                df_filtrado = df[df['valor'] > filter_value]

            # Verificar se o DataFrame está vazio após o filtro
            if df_filtrado.empty:
                st.warning("Nenhum registro encontrado para o filtro selecionado. Filtro não aplicado.")
                return df  # Retorna o DataFrame original para evitar erros em outras partes do código
            else:
                return df_filtrado  # Retorna o DataFrame filtrado
        else:
            return df  # Retorna o DataFrame original se o valor não for válido

def apply_filters_2(df):
    if 'selected_years' in st.session_state:
        df = df[df['data'].dt.year.isin(st.session_state['selected_years'])]
    if 'selected_months' in st.session_state:
        df = df[df['Mês'].isin(st.session_state['selected_months'])]
    if 'credor' in st.session_state:
        df = df[df['credor'].isin(st.session_state['credor'])]
    if 'orgao' in st.session_state:
        df = df[df['orgao'].isin(st.session_state['orgao'])]
    if 'elemento_despesa' in st.session_state:
        df = df[df['Elemento de Despesa'].isin(st.session_state['elemento_despesa'])]
    if 'sub_elemento_despesa' in st.session_state:
        df = df[df['Subelemento'].isin(st.session_state['sub_elemento_despesa'])]
    if 'base_legal' in st.session_state:
        df = df[df['Categorias de base legal'].isin(st.session_state['base_legal'])]
    
    return df

# Função para adicionar CSS personalizado
def add_custom_css():
    st.markdown("""
        <style>
        /* Mudar a cor do fundo e do texto selecionado */
        div[data-baseweb="radio"] > div {
            color: #1E3D59;  /* Cor do texto */
        }
        div[data-baseweb="radio"] > div:hover {
            background-color: #1E3D59;  /* Cor de fundo ao passar o mouse */
            color: white;  /* Cor do texto ao passar o mouse */
        }
        div[data-baseweb="radio"] input:checked + div {
            background-color: #1E3D59;  /* Cor de fundo quando selecionado */
            color: white;  /* Cor do texto quando selecionado */
        }
        </style>
    """, unsafe_allow_html=True)

    
def filters(df_empenho, df_liquidacao, df_pagamento):
    with st.sidebar:
        try:
            st.markdown("## Filtros")
            base = st.radio("Selecione a base de dados", ['Empenhos', 'Liquidação', 'Pagamento'])
            if base == 'Empenhos':
                df = df_empenho
                st.session_state['df_escolhido'] = 'Empenho'
            elif base == 'Liquidação':
                df = df_liquidacao
                st.session_state['df_escolhido'] = 'Liquidação'
            elif base == 'Pagamento':
                df = df_pagamento
                st.session_state['df_escolhido'] = 'Pagamento'
                
            df = year_filter(df)
            df = month_filter(df)
            df = credores(df)
            df = orgao(df)
            df = valores(df)
            
            '''
            df = elemento_despesa(df)
            df = sub_elemento_despesa(df)
            df = Categorias_de_base_legal(df)'''
        
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





# Função para carregar os dados com cache
@st.cache_data
def load_data(file, file_type):
    if file_type == 'json':
        return pd.read_json(file)
    elif file_type == 'csv':
        return pd.read_csv(file)

def itables(df):
    from itables.streamlit import interactive_table

    # Display a DF
    interactive_table(df)

    # Set a caption
    interactive_table(df, caption="A Pandas DataFrame rendered as an interactive DataTable")

    # AddCopy/CSV/Excel download buttons
    interactive_table(df, buttons=["copyHtml5", "csvHtml5", "excelHtml5"])


def rename_columns(df):
    # Dicionário com os nomes das colunas antigos como chaves e os novos nomes como valores
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

    # Renomear as colunas do DataFrame
    df.rename(columns=novos_nomes_colunas, inplace=True)
    
    
    return df

def remove_columns(df, columns):
    return df.drop(columns=columns, errors='ignore')
    

    
def run():
    mkd_text("Prefeitura Municipal de Pojuca - BA", level='title', position='center')
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
                    # Carregar os dados e armazenar no session_state usando cache
                    
                    if file_upload_empenho is not None:
                        
                        df_empenho = load_data(file_upload_empenho, 'json' if file_upload_empenho.name.endswith('.json') else 'csv')
                        if 'df_empenho' in st.session_state:
                            del st.session_state['df_empenho']
                            st.session_state['df_empenho'] = df_empenho
                        else:
                            st.session_state['df_empenho'] = df_empenho
                    
                    if file_upload_liquidacao is not None:
                        df_liquidacao = load_data(file_upload_liquidacao, 'json' if file_upload_liquidacao.name.endswith('.json') else 'csv')
                        if 'df_liquidacao' in st.session_state:
                            del st.session_state['df_liquidacao']
                            st.session_state['df_liquidacao'] = df_liquidacao
                        else:
                            st.session_state['df_liquidacao'] = df_liquidacao
                    
                    if file_upload_pagamento is not None:
                        df_pagamento = load_data(file_upload_pagamento, 'json' if file_upload_pagamento.name.endswith('.json') else 'csv')
                        if 'df_pagamento' in st.session_state:
                            del st.session_state['df_pagamento']
                            st.session_state['df_pagamento'] = df_pagamento
                        else:
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
        columns_to_remove = ['Unidade','Município','Mês','Número do Mês','Código Órgão','Código da Unidade']
        df_to_show = remove_columns(df_to_show, columns_to_remove)
        AgGrid(df_to_show)
        
        
        
        '''tab1, tab2, tab3, tab4 = st.tabs(['Empenhos', 'Liquidação', 'Pagamento', 'Exploração de dados'])
        with tab1:
            df_empenho = st.session_state['df_empenho']
            AgGrid(df_empenho)
        with tab2:
            df_liquidacao = st.session_state['df_liquidacao']
            st.write(df_liquidacao.head())
        with tab3:
            df_pagamento = st.session_state['df_pagamento']
            st.write(df_pagamento.head())
        with tab4:
            option = st.selectbox("Selecione o tipo de dados", ['Empenhos', 'Liquidação', 'Pagamento'])
            if option == 'Empenhos':
                pygwalker(df_empenho)
            elif option == 'Liquidação':
                pygwalker(df_liquidacao)
            else:
                pygwalker(df_pagamento)'''
        
        #itables(df)
        
                
                
                
    '''else:
    
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
    

    metrics(df)'''
    
if __name__ == "__main__":
    st.session_state.clear()
    run()