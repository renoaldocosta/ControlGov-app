import os
import time
import requests
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

from app.services.text_functions import mkd_text, mkd_text_divider

# Dictionary to translate month numbers to Portuguese month names
MONTH_TRANSLATION = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
}


def test_mongodb_connection(db_name: str, collection_name: str) -> MongoClient:
    """
    Test the connection to MongoDB and return the collection.

    Args:
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.

    Returns:
        collection: The MongoDB collection object.

    Raises:
        SystemExit: If unable to connect to the database.
    """
    with st.spinner("Testing MongoDB Connection..."):
        load_dotenv()
        db_password = os.environ.get('db_password')
        uri = f"mongodb+srv://renoaldo_teste:{db_password}@cluster0.zmdkz1p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(uri)

        try:
            client.admin.command('ping')
            
            db = client[db_name]
            collection = db[collection_name]
            return collection
        except Exception as e:
            st.error(f"Error: {e}")
            raise SystemExit("Unable to connect to the database. Please check your URI.")


def format_df(df_empenhos: pd.DataFrame) -> pd.DataFrame:
    """
    Format the 'df_empenhos' DataFrame by cleaning and converting data types.

    Args:
        df_empenhos (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: The formatted DataFrame.
    """
    df = df_empenhos.copy()

    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    if 'Credor' in df.columns:
        df['Credor'] = df['Credor'].str.split(' - ').apply(
            lambda x: f'{x[1]} - {x[0]}' if len(x) > 1 else x[0]
        )

    value_columns = ['Alteração', 'Empenhado', 'Liquidado', 'Pago']
    for column in value_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .str.replace(r'R\$ ?', '', regex=True)
                .str.replace('.', '')
                .str.replace(',', '.')
            )
            df[column] = df[column].astype(float)
        else:
            st.warning(f"Column '{column}' not found in DataFrame.")

    data_columns = ['Data', 'Atualizado']
    for column in data_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column]).dt.date
        else:
            st.warning(f"Column '{column}' not found in DataFrame.")

    return df


def converte_real_float(df, list_columns):
    for column in list_columns:
        new_column = column + '_float'
        df[new_column] = (
            df[column]
            .str.replace('R\$', '', regex=True)  # Remove currency symbol
            .str.replace('.', '', regex=False)   # Remove thousands separator
            .str.replace(',', '.', regex=False)  # Replace decimal commas with dots
            .astype(float)  # Convert to float
        )
    return df  # Fixed return statement


def converte_data_datetime(df, list_columns):
    for column in list_columns:
        new_column = column + '_datetime'
        df[new_column] = pd.to_datetime(df[column], errors='coerce')
    return df

def processa_itens_column(value):
    if isinstance(value, list):
        # Se for uma lista aninhada, converte cada item interno para string e faz join
        return ', '.join(
            ', '.join(map(str, item)) if isinstance(item, list) else str(item)
            for item in value
        )
    return str(value)  # Caso não seja uma lista, apenas converte para string

def controlgov_api_request(url: str) -> pd.DataFrame:
    """
    Make a request to the ControlGov API and return the data as a DataFrame.

    Args:
        url (str): The URL of the API endpoint.

    Returns:
        pd.DataFrame: The data from the API as a DataFrame.
    """
    with st.spinner("Loading data..."):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['empenhos'])
            else:
                st.error(f"Erro ao acessar a API: {response.status_code}")
        except Exception as e:
            st.error(f"Erro ao acessar a API: {e}")
            return pd.DataFrame()
    return df


def split_credor_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'Credor' in df.columns:
        df['Credor'] = df['Credor'].str.split(' - ').apply(
            lambda x: f'{x[1]} - {x[0]}' if len(x) > 1 else x[0]
        )
    return df

def extrai_itens_para_colunas(itens):
    """Extrai a descrição e o valor total dos itens."""
    if isinstance(itens, list) and len(itens) > 1:
        try:
            valores = itens[1][0]  # Acessa a lista interna com os valores
            descricao = valores[0]  # Descrição do item
            valor_total = valores[-1]  # Valor total do item
            return descricao, valor_total
        except IndexError:
            # Caso a estrutura dos dados não esteja correta
            return None, None
    return None, None

@st.cache_data
def get_empenhos_API()-> pd.DataFrame:
    st.session_state.clear()
    with st.spinner("Loading data..."):
        if 'df_empenhos' not in st.session_state:
            url = 'https://api.controlgov.org/empenhos/'
            df_empenhos = controlgov_api_request(url)
            df_empenhos = df_empenhos.astype(str)
            # Remove the 'id' column
            df_empenhos = df_empenhos.drop(['id'], axis=1)
            # Convert date columns to datetime
            colunas_data = ['Data', 'Atualizado']
            df_empenhos = converte_data_datetime(df_empenhos, colunas_data)
            st.write("")
            # Convert multiple currency columns to float
            colunas_valores = ['Alteração', 'Empenhado', 'Liquidado', 'Pago']
            df_empenhos = converte_real_float(df_empenhos, colunas_valores)
            
            
            
            # Extrair descrição e valor dos itens
            if 'Item(ns)' in df_empenhos.columns:
                df_empenhos[['Item_Descricao', 'Item_Valor']] = df_empenhos['Item(ns)'].apply(
                    lambda x: pd.Series(extrai_itens_para_colunas(x))
                )
                df_empenhos = df_empenhos.drop(columns=['Item(ns)'])  # Remove coluna antiga

            df_empenhos = split_credor_column(df_empenhos)
            
            st.session_state['df_empenhos'] = df_empenhos
            
            # renomear colunas 
            #df_empenhos.rename(columns={'Elemento_de_Despesa': 'Elemento de Despesa','Categorias_de_base_legal':'Categorias de base legal'}, inplace=True)
            return df_empenhos
        else:
            return st.session_state['df_empenhos']


@st.cache_data
def get_empenhos(db_name: str, collection_name: str) -> pd.DataFrame:
    """
    Retrieve 'empenhos' data from MongoDB and return it as a DataFrame.

    Args:
        db_name (str): The name of the MongoDB database.
        collection_name (str): The name of the collection.

    Returns:
        pd.DataFrame: The 'empenhos' data as a DataFrame.
    """
    with st.spinner("Loading data..."):
        mongodb_collection = test_mongodb_connection(db_name, collection_name)
        if 'df_empenhos' not in st.session_state:
            df_empenhos = pd.DataFrame(list(mongodb_collection.find()))

            if 'Item(ns)' in df_empenhos.columns:
                df_empenhos['Item(ns)'] = df_empenhos['Item(ns)'].apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )

            df_empenhos = format_df(df_empenhos)
            return df_empenhos
        else:
            return st.session_state['df_empenhos']


def metrics(df: pd.DataFrame):
    """
    Display metrics of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame for which to calculate metrics.
    """
    if 'df_escolhido' in st.session_state:
        df_escolhido = st.session_state['df_escolhido']
        mkd_text_divider(f"Métricas de {df_escolhido}", level='subheader', position='center')
    else:
        mkd_text_divider("Métricas de Empenho", level='subheader', position='center')
    
    if df['Data'].dtype != 'datetime64[ns]':
        df['Data'] = pd.to_datetime(df['Data'])
    
    total_registros = df.shape[0]
    data_mais_recente = df['Data_datetime'].max().strftime('%d/%m/%Y')
    data_mais_antiga = df['Data_datetime'].min().strftime('%d/%m/%Y')

    valor_minimo = df['Empenhado_float'].min()
    valor_medio = df['Empenhado_float'].mean()
    valor_maximo = df['Empenhado_float'].max()
    
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


def format_currency(value: float, currency_symbol: str = '') -> str:
    """
    Format a numeric value to Brazilian currency format.

    Args:
        value (float): Numeric value.
        currency_symbol (str, optional): Currency symbol to prepend. Defaults to ''.

    Returns:
        str: Value formatted as Brazilian currency.
    """
    if value != '' and not pd.isnull(value):
        formatted_value = f"{value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        if currency_symbol:
            return f"{currency_symbol} {formatted_value}"
        else:
            return formatted_value
    else:
        if currency_symbol:
            return f"{currency_symbol} 0,00"
        else:
            return "0,00"


def apply_currency_format(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply Brazilian currency formatting to specific columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the columns to be formatted.
        columns (list): List of columns to apply currency formatting.

    Returns:
        pd.DataFrame: DataFrame with specified columns formatted as Brazilian currency.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(format_currency)
        else:
            st.warning(f"The column '{column}' is not present in the DataFrame.")
    return df


def year_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on a selected range of years.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    first_year = df['Data'].dt.year.min()
    last_year = df['Data'].dt.year.max()

    selected_years = st.slider(
        "Ano",
        min_value=int(first_year),
        max_value=int(last_year),
        value=(int(first_year), int(last_year))
    )

    df_filtered = df[df['Data'].dt.year.between(*selected_years)]
    st.session_state['selected_years'] = selected_years

    return df_filtered


def month_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected months.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df['Mês_Numero'] = df['Data'].dt.month
    df['Mês'] = df['Mês_Numero'].map(MONTH_TRANSLATION)
    all_months = list(MONTH_TRANSLATION.values())

    months = st.multiselect("Mês", all_months, placeholder="Selecione um ou mais meses")

    if months:
        df = df[df['Mês'].isin(months)]

    df = df.sort_values(by='Mês_Numero')
    st.session_state['selected_months'] = months

    return df


def credores_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected creditors.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    credores_list = df['Credor'].unique().tolist()

    selected_creditors = st.multiselect(
        "Credores / Fornecedores",
        credores_list,
        placeholder="Selecione um ou mais credores"
    )

    if selected_creditors:
        df = df[df['Credor'].isin(selected_creditors)]

    st.session_state['credores'] = df

    return df


def elemento_despesa_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected 'Elemento de Despesa'.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    elementos = df['Elemento_de_Despesa'].unique().tolist()
    selected_elementos: list[str] = st.multiselect(
        "Elemento de Despesa",
        elementos,
        placeholder="Selecione um ou mais elementos de despesa"
    )

    if selected_elementos:
        df = df[df['Elemento_de_Despesa'].isin(selected_elementos)]

    st.session_state['elemento_despesa'] = df

    return df


def sub_elemento_despesa_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected 'Subelemento'.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    sub_elementos = sorted(df['Subelemento'].unique().tolist())

    selected_sub_elementos = st.multiselect(
        "Subelemento de Despesa",
        sub_elementos,
        placeholder="Selecione um ou mais subelementos de despesa"
    )

    if selected_sub_elementos:
        df = df[df['Subelemento'].isin(selected_sub_elementos)]

    st.session_state['sub_elemento_despesa'] = df

    return df


def categorias_de_base_legal_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame based on selected 'Categorias de base legal'.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    base_legal_list = sorted(df['Categorias_de_base_legal'].unique().tolist())

    selected_base_legal = st.multiselect(
        "Base Legal",
        base_legal_list,
        placeholder="Selecione uma ou mais bases legais"
    )

    if selected_base_legal:
        df = df[df['Categorias_de_base_legal'].isin(selected_base_legal)]

    st.session_state['base_legal'] = df

    return df


def filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multiple filters to the DataFrame based on user selections.

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    with st.sidebar:
        try:
            with st.expander("## Filtros", expanded=True):
                level = 'h4'
                mkd_text("Período", level=level, position='center')
                df = year_filter(df)
                df = month_filter(df)

                mkd_text("Outros Filtros", level=level, position='center')
                df = credores_filter(df)
                df = elemento_despesa_filter(df)
                df = sub_elemento_despesa_filter(df)
                df = categorias_de_base_legal_filter(df)

                return df
        except Exception as e:
            st.error(f"Erro ao aplicar filtros: {e}")
            return df


def remove_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove unwanted columns from the DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame.
        columns (list): List of columns to be removed.

    Returns:
        pd.DataFrame: DataFrame without the specified columns.
    """
    return df.drop(columns=columns, errors='ignore')


def display_aggrid_with_links(
    df: pd.DataFrame,
    link_columns: list,
    link_text: dict = None,
    right_align_columns: list = None,
    height: int = 300,
    theme: str = 'balham',
    update_mode: GridUpdateMode = GridUpdateMode.NO_UPDATE
):
    """
    Display a DataFrame in AgGrid with specified columns rendered as clickable links and custom alignment.

    Args:
        df (pd.DataFrame): The DataFrame to display.
        link_columns (list): List of column names that contain URLs to be rendered as links.
        link_text (dict, optional): Dictionary where keys are link column names and values are link display texts.
                                    If not provided, the default text will be 'Detalhes'.
        right_align_columns (list, optional): List of column names to be right-aligned.
        height (int, optional): Height of the AgGrid table. Defaults to 300.
        theme (str, optional): Theme for AgGrid. Defaults to 'balham'.
        update_mode (GridUpdateMode, optional): Grid update mode. Defaults to GridUpdateMode.NO_UPDATE.
    """
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, flex=1)

    for col in link_columns:
        current_link_text = link_text.get(col, 'Detalhes') if link_text else 'Detalhes'

        cell_renderer = JsCode(f"""
            class UrlCellRenderer {{
                init(params) {{
                    this.eGui = document.createElement('a');
                    this.eGui.innerText = '{current_link_text}';
                    this.eGui.setAttribute('href', params.value);
                    this.eGui.setAttribute('target', '_blank');
                    this.eGui.style.color = '#1a0dab';
                    this.eGui.style.textDecoration = 'none';
                }}
                getGui() {{
                    return this.eGui;
                }}
            }}
        """)

        gb.configure_column(
            col,
            headerName=col.replace('_', ' ').title(),
            cellRenderer=cell_renderer,
            width=150,
            suppressSizeToFit=True
        )

    if right_align_columns:
        for col in right_align_columns:
            gb.configure_column(
                col,
                headerName=col.replace('_', ' ').title(),
                cellStyle={'text-align': 'right'},
                type=["numericColumn"]
            )

    gridOptions = gb.build()

    AgGrid(
        df,
        gridOptions=gridOptions,
        update_mode=update_mode,
        height=height,
        theme=theme,
        allow_unsafe_jscode=True
    )


def display_dataframe_with_links(
    df: pd.DataFrame,
    link_columns: list,
    link_text: dict = None,
    right_align_columns: list = None,
    height: int = 300,
    theme: str = 'default',
    use_data_editor: bool = False
):
    """
    Display a DataFrame in Streamlit with specified columns rendered as clickable links and custom alignment.

    Args:
        df (pd.DataFrame): The DataFrame to display.
        link_columns (list): List of column names that contain URLs to be rendered as links.
        link_text (dict, optional): Dictionary where keys are link column names and values are link display texts.
                                    If not provided, the default text will be the full URL.
        right_align_columns (list, optional): List of column names to be right-aligned.
        height (int, optional): Height of the table in Streamlit. Defaults to 300.
        theme (str, optional): Theme for the component (for data_editor). Defaults to 'default'.
        use_data_editor (bool, optional): If True, uses st.data_editor; otherwise, uses st.dataframe. Defaults to False.
    """
    column_config = {}

    for col in link_columns:
        column_config[col] = st.column_config.LinkColumn(
            label=col.replace('_', ' ').title(),
            display_text=link_text.get(col, None) if link_text else None,
            width="large",
            help=f"Link para {col.replace('_', ' ').lower()}",
            validate=r"^https?://.+$"
        )

    if right_align_columns:
        for col in right_align_columns:
            column_config[col] = st.column_config.NumberColumn(
                label=col.replace('_', ' ').title(),
                format="%.2f",
                width="small",
                help=f"Valores para {col.replace('_', ' ').lower()}"
            )

    if use_data_editor:
        st.data_editor(
            df,
            column_config=column_config,
            hide_index=True,
            height=height,
            use_container_width=True
        )
    else:
        st.dataframe(
            df,
            column_config=column_config,
            hide_index=True,
            height=height,
            use_container_width=True
        )


def reorder_columns(df: pd.DataFrame, new_order: list) -> pd.DataFrame:
    """
    Reorder the columns of a DataFrame based on a provided new order.

    Args:
        df (pd.DataFrame): The original DataFrame.
        new_order (list): List specifying the new order of columns.

    Returns:
        pd.DataFrame: DataFrame with columns reordered.
    """
    missing_cols = [col for col in new_order if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")

    return df[new_order]


def format_data_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Format date columns in the DataFrame to 'dd/mm/yyyy' string format.

    Args:
        df (pd.DataFrame): DataFrame containing date columns.
        columns (list): List of date columns to format.

    Returns:
        pd.DataFrame: DataFrame with formatted date columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%d/%m/%Y')
        else:
            st.warning(f"Column '{col}' not found in DataFrame.")
    return df


def run():
    """
    Main function to run the Streamlit app.
    """
    mkd_text("Câmara Municipal de Pinhão - SE", level='title', position='center')

    # db_name, collection_name = 'CMP', 'EMPENHOS_DETALHADOS_STAGE'
    # df_empenhos = get_empenhos(db_name, collection_name)
    
    df_empenhos = get_empenhos_API()
    st.dataframe(df_empenhos)

    df_filtered = filters(df_empenhos)
    metrics(df_filtered)
    mkd_text_divider("Registros", level='subheader', position='center')

    tab1, tab2 = st.tabs(['Empenhos', 'Exploração'])

    df_to_show = df_filtered.copy().sort_values(by='Data', ascending=False)
    columns_to_remove = [
        'Poder', 'Função', "Subfunção", "Item(ns)", 'Mês', 'Mês_Numero',
        'Unid. Administradora', 'Unid. Orçamentária', 'Fonte de recurso'
    ]
    df_to_show = remove_columns(df_to_show, columns_to_remove)

    new_column_order = [
        "Número", "Data", "Subelemento", "Credor", "Alteração", "Empenhado",
        "Liquidado", "Pago", "Atualizado", "link_Detalhes", "Elemento de Despesa",
        "Projeto/Atividade", "Categorias de base legal", "Histórico"
    ]
    df_to_show = reorder_columns(df_to_show, new_column_order)

    date_columns = ['Data', 'Atualizado']
    df_to_show = format_data_columns(df_to_show, date_columns)

    currency_columns = ['Empenhado', 'Liquidado', 'Pago', 'Alteração']
    df_to_show = apply_currency_format(df_to_show, currency_columns)

    link_cols = ["link_Detalhes"]
    link_texts = {"link_Detalhes": "Ver Detalhes"}
    right_align_cols = ['Empenhado', 'Liquidado', 'Pago', 'Alteração']

    with tab1:
        display_dataframe_with_links(
            df=df_to_show,
            link_columns=link_cols,
            link_text=link_texts,
            right_align_columns=right_align_cols,
            height=300,
            theme='default',
            use_data_editor=False
        )

    with tab2:
        display_aggrid_with_links(
            df=df_to_show,
            link_columns=link_cols,
            link_text=link_texts,
            right_align_columns=right_align_cols,
            height=300,
            theme='balham',
            update_mode=GridUpdateMode.NO_UPDATE
        )


if __name__ == "__main__":
    run()
