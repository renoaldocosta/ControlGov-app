import streamlit as st
from app.services.text_functions import mkd_text_divider, mkd_text, mkd_paragraph
import pandas as pd
import pandas as pd
import os
import time
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
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
    with st.spinner("Testando Conexão..."):
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
        df[column] = pd.to_datetime(df[column]).dt.date
        
    return df


@st.cache_data
def get_empenhos(db, collection):
    with st.spinner("Carregando dados..."):
        progress_text = "Operação em Progresso. Por favor aguarde."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        
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
    """
    Exibe métricas do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame para cálculo das métricas.
    """
    if 'df_escolhido' in st.session_state:
        df_escolhido = st.session_state['df_escolhido']
        mkd_text_divider(f"Métricas de {df_escolhido}", level='subheader', position='center')
    else:
        mkd_text_divider("Métricas de Empenho", level='subheader', position='center')
        
    # Converter colunas de data para o tipo datetime
    df['Data'] = pd.to_datetime(df['Data'])
    df['Data'] = pd.to_datetime(df['Data'])

    # Cálculo das métricas
    total_registros = df.shape[0]
    data_mais_recente = df['Data'].max().strftime('%d/%m/%Y')  # Converte a data para string
    data_mais_antiga = df['Data'].min().strftime('%d/%m/%Y')    # Converte a data para string
    valor_minimo = df['Empenhado'].min()
    valor_medio = df['Empenhado'].mean()
    valor_maximo = df['Empenhado'].max()

    # Exibir métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", total_registros)
    with col2:
        st.metric("Data Mais Antiga", data_mais_antiga)
    with col3:
        st.metric("Data Mais Recente", data_mais_recente)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Valor Mínimo", format_currency(valor_minimo, currency_symbol='R$'))
    with col5:
        st.metric("Valor Médio", format_currency(valor_medio, currency_symbol='R$'))
    with col6:
        st.metric("Valor Máximo", format_currency(valor_maximo, currency_symbol='R$'))



def format_currency(value, currency_symbol='') -> str:
    """
    Formata um valor numérico para o estilo brasileiro de moeda.

    Args:
        value (float): Valor numérico.

    Returns:
        str: Valor formatado como moeda brasileira.
    """
    if currency_symbol:
        if value != '':
            return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        else:
            return "R$ 0,00"
    else:
        if value != '':
            return f"{value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        else:
            return "0,00"

def apply_currency_format(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Aplica a formatação de moeda brasileira a colunas específicas de um DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas a serem formatadas.
        columns (list): Lista de colunas para aplicar a formatação de moeda.

    Returns:
        pd.DataFrame: DataFrame com as colunas formatadas como moeda brasileira.
    """
    for column in columns:
        if column in df.columns:
            # Aplica a função format_currency a cada valor da coluna
            df[column] = df[column].apply(format_currency)
        else:
            raise ValueError(f"A coluna '{column}' não está presente no DataFrame.")
    
    return df

def metrics_1(df):
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


def filters(df):
    with st.sidebar:
        try:
            with st.expander("## Filtros", expanded=True):
                    
                df = df
                level = 'h4'
                mkd_text("Período", level=level, position='center')
                df = year_filter(df)
                df = month_filter(df)
                
                mkd_text("Outros Filtros", level=level, position='center')
                df = credores(df)
                df = elemento_despesa(df)
                df = sub_elemento_despesa(df)
                df = Categorias_de_base_legal(df)
                
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

def remove_columns(df, columns):
    """
    Remove colunas indesejadas do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.
        columns (list): Lista de colunas a serem removidas.

    Returns:
        pd.DataFrame: DataFrame sem as colunas especificadas.
    """
    return df.drop(columns=columns, errors='ignore')


def display_aggrid_with_links(
    df: pd.DataFrame,
    link_columns: list,
    link_text: dict = None,
    right_align_columns: list = None,  # Novo parâmetro para alinhamento à direita
    height: int = 300,
    theme: str = 'balham',
    update_mode: GridUpdateMode = GridUpdateMode.NO_UPDATE
):
    """
    Exibe um DataFrame no AgGrid com colunas específicas renderizadas como links clicáveis e com alinhamento personalizado.

    Parameters:
    - df (pd.DataFrame): DataFrame a ser exibido.
    - link_columns (list): Lista de nomes de colunas que contêm URLs.
    - link_text (dict, opcional): Dicionário onde as chaves são os nomes das colunas de links
                                   e os valores são os textos dos links. Se não fornecido,
                                   o texto padrão será 'Detalhes'.
    - right_align_columns (list, opcional): Lista de nomes de colunas que devem ser alinhadas à direita.
    - height (int, opcional): Altura da tabela AgGrid. Padrão é 300.
    - theme (str, opcional): Tema do AgGrid. Padrão é 'balham'.
    - update_mode (GridUpdateMode, opcional): Modo de atualização do grid. Padrão é GridUpdateMode.NO_UPDATE.
    """
    
    # Inicializa o GridOptionsBuilder a partir do DataFrame
    gb = GridOptionsBuilder.from_dataframe(df)
    
    # Configura as colunas padrão
    gb.configure_default_column(resizable=True, flex=1)  # Permitir redimensionamento
    
    # Itera sobre as colunas de links e configura o cellRenderer para cada uma
    for col in link_columns:
        # Define o texto do link: usa o valor de link_text se fornecido, senão 'Detalhes'
        current_link_text = link_text.get(col, 'Detalhes') if link_text else 'Detalhes'
        
        # Define o cellRenderer personalizado usando JsCode
        cell_renderer = JsCode(f"""
            class UrlCellRenderer {{
                init(params) {{
                    this.eGui = document.createElement('a');
                    this.eGui.innerText = '{current_link_text}';
                    this.eGui.setAttribute('href', params.value);
                    this.eGui.setAttribute('target', '_blank');
                    this.eGui.style.color = '#1a0dab';  // Cor do link
                    this.eGui.style.textDecoration = 'none';  // Remove sublinhado
                }}
                getGui() {{
                    return this.eGui;
                }}
            }}
        """)
        
        # Configura a coluna com o cellRenderer
        gb.configure_column(
            col,
            headerName=col.replace('_', ' ').title(),  # Formata o nome do cabeçalho
            cellRenderer=cell_renderer,
            width=150,
            suppressSizeToFit=True
        )
    
    # Se houver colunas para alinhar à direita, configura o estilo delas
    if right_align_columns:
        for col in right_align_columns:
            gb.configure_column(
                col,
                headerName=col.replace('_', ' ').title(),
                cellStyle={'text-align': 'right'},  # Alinha o conteúdo à direita
                type=["numericColumn"]  # Opcional: adiciona tipo numérico para melhor formatação
            )
    
    # Constrói as opções do grid
    gridOptions = gb.build()
    
    # Renderiza o AgGrid no Streamlit
    AgGrid(
        df,
        gridOptions=gridOptions,
        update_mode=update_mode,
        height=height,
        theme=theme,
        allow_unsafe_jscode=True  # Necessário para permitir o uso de JsCode
    )

def reorder_columns(df: pd.DataFrame, new_order: list) -> pd.DataFrame:
    """
    Reordena as colunas de um DataFrame com base em uma nova ordem fornecida.

    Parameters:
    - df (pd.DataFrame): O DataFrame original.
    - new_order (list): Lista com a nova ordem das colunas.

    Returns:
    - pd.DataFrame: DataFrame com as colunas reorganizadas.
    """
    # Verifica se todas as colunas da nova ordem existem no DataFrame
    missing_cols = [col for col in new_order if col not in df.columns]
    if missing_cols:
        raise ValueError(f"As seguintes colunas estão ausentes no DataFrame: {missing_cols}")
    
    # Reordena o DataFrame com base na nova ordem de colunas
    return df.reindex(columns=new_order)

def format_data_columns(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col]).dt.strftime('%d/%m/%Y')
    return df
        
        
def run():
    
    
    mkd_text("Câmara Municipal de Pinhão - SE", level='title', position='center')
    # Define database and collection
    db, collection = 'CMP', 'EMPENHOS_DETALHADOS_STAGE'
    
    
        # Test connection to MongoDB
    mongodb_collection = testa_conexao_mongodb(db, collection)
    
        
    # Retrieve data from MongoDB as a DataFrame
    df_empenhos = get_empenhos(db, collection)
    st.session_state['df_empenhos'] = df_empenhos
    df = filters(df_empenhos)
    metrics(df)
    mkd_text_divider("Registros", level='subheader', position='center')
    
    tab1, tab2 = st.tabs(['Empenhos', 'Exploração'])
    
    
    
    with tab1:
        df_to_show = df.copy().sort_values(by='Data', ascending=False)
        columns_to_remove = ['Poder', 'Função',"Subfunção","Item(ns)", 'Mês', 'Mês_Numero', 'Unid. Administradora','Unid. Orçamentária', 'Fonte de recurso']
        df_to_show = remove_columns(df_to_show, columns_to_remove)
        
        # Nova ordem das colunas
        new_column_order = [
            "Número", "Data", "Subelemento", "Credor", "Alteração", "Empenhado",
            "Liquidado", "Pago", "Atualizado", "link_Detalhes", "Elemento de Despesa",
            "Projeto/Atividade", "Categorias de base legal", "Histórico"
        ]

        # Chama a função para reordenar as colunas
        df_to_show = reorder_columns(df_to_show, new_column_order)

        date_columns = ['Data', 'Atualizado']
        df_to_show = format_data_columns(df_to_show, date_columns)
        
        # Lista de colunas para aplicar a formatação
        currency_columns = ['Empenhado', 'Liquidado', 'Pago', 'Alteração']

        # Aplica a função para formatar as colunas de moeda
        df_to_show = apply_currency_format(df_to_show, currency_columns)


        # Grid
        # Define quais colunas são links
        link_cols = ["link_Detalhes"]

        # Define textos personalizados para os links (opcional)
        link_texts = {
            "link_Detalhes": "Ver Detalhes",
        }

        # Define as colunas que devem ser alinhadas à direita
        right_align_cols = ['Empenhado', 'Liquidado', 'Pago', 'Alteração']

        # Chama a função para exibir o AgGrid com links e alinhamento personalizado
        display_aggrid_with_links(
            df=df_to_show,
            link_columns=link_cols,
            link_text=link_texts,  # Você pode omitir este parâmetro se quiser usar o texto padrão 'Detalhes'
            right_align_columns=right_align_cols,  # Passa as colunas para alinhar à direita
            height=300,
            theme='balham',
            update_mode=GridUpdateMode.NO_UPDATE
        )
    with tab2:
        pygwalker(df_empenhos)

    
    
if __name__ == "__main__":
    run()