import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")

# =========================
# 0) FUNÇÃO create_refined ATUALIZADA
# =========================

def create_refined(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame de transações (sem exigir 'is_fraud' ou 'estado')
    e retorna um DataFrame 'refined' contendo as features temporais, de deslocamento,
    e preenche 'estado' com 'sp' para cada linha.
    """
    # 1) Carrega os dados de pagadores e vendedores
    df_payers = pd.read_feather('data/payers_raw.feather')
    df_sellers = pd.read_feather('data/seller_raw.feather')

    # 2) Merge com pagadores
    df = df_transactions.merge(
        df_payers,
        left_on='card_id',
        right_on='card_hash',
        how='left',
        suffixes=('', '_payer')
    )

    # 3) Merge com vendedores
    df = df.merge(
        df_sellers,
        on='terminal_id',
        how='left',
        suffixes=('', '_seller')
    )

    # 4) Seleção das colunas necessárias para o raw_merge (sem 'is_fraud' e sem 'estado')
    desired = [
        'transaction_id',
        'tx_datetime',
        'tx_date',
        'tx_time',
        'tx_amount',
        'card_id',
        'card_bin',
        'card_first_transaction',
        'terminal_id',
        'latitude',
        'longitude',
        'terminal_operation_start',
        'terminal_soft_descriptor'
    ]
    df_raw_merge = df[desired].copy()

    # 5) Conversão de latitude/longitude para float
    df_raw_merge['latitude']  = df_raw_merge['latitude'].astype(float)
    df_raw_merge['longitude'] = df_raw_merge['longitude'].astype(float)

    # 6) Conversão de datas e extração de features temporais
    df_raw_merge['tx_datetime']              = pd.to_datetime(df_raw_merge['tx_datetime'])
    df_raw_merge['card_first_transaction']   = pd.to_datetime(df_raw_merge['card_first_transaction'])
    df_raw_merge['terminal_operation_start'] = pd.to_datetime(df_raw_merge['terminal_operation_start'])

    df_raw_merge['tx_hour'] = df_raw_merge['tx_datetime'].dt.hour
    df_raw_merge['tx_dow']  = df_raw_merge['tx_datetime'].dt.dayofweek

    df_raw_merge['card_age_days'] = (
        df_raw_merge['tx_datetime'] - df_raw_merge['card_first_transaction']
    ).dt.days
    df_raw_merge['terminal_age_days'] = (
        df_raw_merge['tx_datetime'] - df_raw_merge['terminal_operation_start']
    ).dt.days

    # 7) Histórico do cartão: valor e intervalo entre transações
    df_raw_merge = df_raw_merge.sort_values(['card_id', 'tx_datetime'])
    df_raw_merge['tx_amount_prev'] = (
        df_raw_merge.groupby('card_id')['tx_amount']
        .shift(1)
        .fillna(0)
    )
    df_raw_merge['hours_since_prev'] = (
        df_raw_merge['tx_datetime']
        - df_raw_merge.groupby('card_id')['tx_datetime'].shift(1)
    ).dt.total_seconds().div(3600).fillna(0)

    # 8) Cálculo de travel_speed via fórmula de Haversine
    df_raw_merge['prev_lat'] = df_raw_merge.groupby('card_id')['latitude'].shift(1)
    df_raw_merge['prev_lon'] = df_raw_merge.groupby('card_id')['longitude'].shift(1)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Raio da Terra em km
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = (np.sin(dphi / 2)**2
             + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2)
        return 2 * R * np.arcsin(np.sqrt(a))

    df_raw_merge['distance_km'] = haversine(
        df_raw_merge['prev_lat'],
        df_raw_merge['prev_lon'],
        df_raw_merge['latitude'],
        df_raw_merge['longitude']
    )
    df_raw_merge['travel_time_h'] = df_raw_merge['hours_since_prev'].replace(0, np.nan)
    df_raw_merge['speed_kmh']     = df_raw_merge['distance_km'] / df_raw_merge['travel_time_h']
    df_raw_merge['travel_speed']  = df_raw_merge['speed_kmh']

    # 9) Preenche 'estado' com valor fixo 'sp'
    df_raw_merge['estado'] = 'sp'

    # 10) Seleção das colunas finais para o "refined" (incluindo 'estado')
    features = [
        'tx_datetime',
        'tx_amount',
        'tx_amount_prev',
        'hours_since_prev',
        'tx_hour',
        'tx_dow',
        'card_age_days',
        'terminal_age_days',
        'card_bin',
        'latitude',
        'longitude',
        'travel_speed',
        'estado'
    ]
    df_refined = df_raw_merge[features].reset_index(drop=True)

    return df_refined


# =========================
# 1) CONFIGURAÇÕES DE MODELO (no topo, em funções)
# =========================

# 📌 Defina aqui os valores padrão para o Tracking URI e Run ID do MLflow:
MLFLOW_TRACKING_URI = "http://18.231.44.164:8080"
RUN_ID = "bc390b0166cb4d22ba1a7d3ef0e30a1d"

# 🎯 Threshold padrão para classificar como fraude (ajuste se necessário)
DEFAULT_THRESHOLD = 0.50


def configure_mlflow(uri: str):
    """
    Configura o MLflow para usar o Tracking Server especificado.
    Basta chamar configure_mlflow(MLFLOW_TRACKING_URI) antes de carregar o modelo.
    """
    mlflow.set_tracking_uri(uri)


@st.cache_resource
def load_model_from_mlflow(run_id: str):
    """
    Carrega o modelo salvo no MLflow pelo run_id.
    Retorna (model, True) se carregou com sucesso, ou (None, False) em caso de erro.
    Internamente, espera que o modelo esteja registrado em "runs:/<run_id>/pipeline-final".
    """
    modelo = None
    sucesso = False
    try:
        model_uri = f"runs:/{run_id}/pipeline-final"
        modelo = mlflow.sklearn.load_model(model_uri)
        sucesso = True
    except Exception:
        sucesso = False
        modelo = None
    return modelo, sucesso


# =========================
# 2) FUNÇÃO DE CÁLCULO DE MÉTRICAS DE NEGÓCIO
# =========================

def business_profit_metric(y_true: np.ndarray, y_pred: np.ndarray, amount: np.ndarray):
    """
    Calcula o lucro baseado na métrica de negócio:
    - 3% de lucro em transações legítimas detectadas corretamente
    - 100% de prejuízo em fraudes não detectadas
    """
    mask_true_negative = (y_true == 0) & (y_pred == 0)
    mask_false_negative = (y_true == 1) & (y_pred == 0)

    lucro_legitimas = (amount[mask_true_negative] * 0.03).sum()
    prejuizo_fraudes = amount[mask_false_negative].sum()
    lucro_total = lucro_legitimas - prejuizo_fraudes

    return {
        "lucro_total": lucro_total,
        "lucro_legitimas": lucro_legitimas,
        "prejuizo_fraudes": prejuizo_fraudes,
        "num_legitimas_corretas": mask_true_negative.sum(),
        "num_fraudes_perdidas": mask_false_negative.sum(),
    }


# =========================
# 3) CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="MercadoPago – Detecção de Fraudes",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# 4) CSS PERSONALIZADO (pinta o fundo e estiliza botões)
# =========================

st.markdown(
    """
    <style>
        /* Esconde menu e rodapé padrão do Streamlit */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Pinta todo o fundo da aplicação */
        .css-18e3th9 {
            background-color: #f8f9fa !important;
        }
        .css-1d391kg {
            background-color: #f8f9fa !important;
        }

        /* Estiliza botões com gradiente */
        .stButton > button {
            background: linear-gradient(90deg, #009ee3 0%, #0073e6 100%);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 158, 227, 0.4);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 5) HEADER (logo + título)
# =========================

st.markdown(
    """
    <div style="background: linear-gradient(90deg, #009ee3 0%, #0073e6 100%);
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 2.5rem; font-weight: bold; margin: 0; text-align: center;">
            💳 MercadoPago
        </h1>
        <p style="color: #e3f2fd; font-size: 1.2rem; text-align: center; margin: 0;">
            Sistema de Detecção de Fraudes
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# 6) CONFIGURAÇÃO DO MLflow E CARREGAMENTO DO MODELO
# =========================

# Configura o tracking URI para o MLflow
configure_mlflow(MLFLOW_TRACKING_URI)

# Carrega o modelo uma única vez (cacheado)
model, model_loaded = load_model_from_mlflow(RUN_ID)

# =========================
# 7) LAYOUT PRINCIPAL EM COLUNAS
# =========================

col_config, col_analysis = st.columns([1, 3], gap="large")

# ----- Coluna de CONFIGURAÇÕES (upload de dataset e aviso de modelo) -----
with col_config:
    st.markdown("## 🔧 Configurações")
    st.markdown("---")

    # Exibe status do modelo automaticamente carregado
    if model_loaded:
        st.success("✅ Modelo MLflow carregado")
    else:
        st.error("❌ Falha ao carregar modelo")
        st.info(f"Verifique se o run ID '{RUN_ID}' está correto ou se o MLflow URI está acessível.")

    st.markdown("---")

    # Upload do dataset (permanece em configurações)
    st.markdown("#### 📊 Dataset para Análise")
    uploaded_file = st.file_uploader(
        "📁 Carregue o dataset (.feather)",
        type=["feather"],
        help="Dataset com transações para análise",
    )
    if uploaded_file:
        try:
            df_preview = pd.read_feather(uploaded_file)
            st.success(f"✅ {len(df_preview):,} transações encontradas")
            if "is_fraud" in df_preview.columns:
                fraud_count = df_preview["is_fraud"].sum()
                st.info(f"🎯 {fraud_count:,} fraudes reais")
            else:
                st.warning("⚠️ Coluna 'is_fraud' não encontrada")
        except Exception:
            st.error("❌ Erro ao ler o arquivo .feather")

    st.markdown("---")
    st.caption(f"Modelo obtido de: runs:/{RUN_ID}/pipeline-final")
    st.caption(f"Tracking URI: {MLFLOW_TRACKING_URI}")

# ----- Coluna de ANÁLISE (tudo que envolve métricas, gráficos e predição) -----
with col_analysis:
    if (uploaded_file is not None) and model_loaded:
        try:
            df_raw = pd.read_feather(uploaded_file)

            st.markdown("### 📋 Informações do Dataset")
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.markdown(
                    f"""
                    <div style="background: white; 
                                padding: 1.5rem; 
                                border-radius: 10px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                border-left: 4px solid #009ee3; 
                                margin-bottom: 1rem;">
                        <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                            {len(df_raw):,}
                        </div>
                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                            Total de Transações
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_i2:
                if "is_fraud" in df_raw.columns:
                    fraud_count = df_raw["is_fraud"].sum()
                    fraud_rate = (fraud_count / len(df_raw)) * 100
                    st.markdown(
                        f"""
                        <div style="background: white; 
                                    padding: 1.5rem; 
                                    border-radius: 10px; 
                                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                    border-left: 4px solid #009ee3; 
                                    margin-bottom: 1rem;">
                            <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                {fraud_count:,}
                            </div>
                            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                Fraudes Reais ({fraud_rate:.2f}%)
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: white; 
                                    padding: 1.5rem; 
                                    border-radius: 10px; 
                                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                    border-left: 4px solid #009ee3; 
                                    margin-bottom: 1rem;">
                            <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                N/A
                            </div>
                            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                Fraudes (sem labels)
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_i3:
                if "tx_amount" in df_raw.columns:
                    total_amount = df_raw["tx_amount"].sum()
                    avg_amount = df_raw["tx_amount"].mean()
                    st.markdown(
                        f"""
                        <div style="background: white; 
                                    padding: 1.5rem; 
                                    border-radius: 10px; 
                                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                    border-left: 4px solid #009ee3; 
                                    margin-bottom: 1rem;">
                            <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                R$ {total_amount:,.2f}
                            </div>
                            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                Volume Total
                            </div>
                            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">
                                Média: R$ {avg_amount:.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # -----------------------------------
            # Botão para executar análise de fraude
            # -----------------------------------
            if st.button("🚀 Executar Análise de Fraudes"):
                with st.spinner("↪️ Formatando dados, realizando predições e calculando métricas..."):
                    try:
                        # 1) Separa y_true (se existir) e remove de df_raw
                        if "is_fraud" in df_raw.columns:
                            y_true = df_raw["is_fraud"].values
                            df_raw = df_raw.drop(columns=["is_fraud"])
                        else:
                            y_true = None

                        # 2) Chama a função que gera o df_refined (inclui 'estado'='sp')
                        df_refined = create_refined(df_raw)

                        # 3) Prepara X_test
                        X_test = df_refined.copy()

                        # 4) Predição de probabilidade
                        if hasattr(model, "predict_proba"):
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                        else:
                            y_pred_binary = model.predict(X_test)
                            y_pred_proba = y_pred_binary.astype(float)

                        # 5) Converte para binário
                        y_pred = (y_pred_proba >= DEFAULT_THRESHOLD).astype(int)

                        # 6) Valores para métricas financeiras
                        amounts = (
                            X_test["tx_amount"].values
                            if "tx_amount" in X_test.columns
                            else np.ones(len(X_test))
                        )

                        # 7) Se existirem labels, calculamos métricas de negócio
                        if y_true is not None:
                            business_metrics = business_profit_metric(
                                y_true, y_pred, amounts
                            )

                            st.markdown("### 💰 Resultados Financeiros")
                            col_r1, col_r2, col_r3 = st.columns(3, gap="large")

                            with col_r1:
                                profit_style = (
                                    "color: #4caf50;"
                                    if business_metrics["lucro_total"] >= 0
                                    else "color: #f44336;"
                                )
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; {profit_style}">
                                            R$ {business_metrics["lucro_total"]:,.2f}
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Lucro/Prejuízo Total
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with col_r2:
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; color: #4caf50;">
                                            R$ {business_metrics["lucro_legitimas"]:,.2f}
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Lucro de Legítimas
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with col_r3:
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; color: #f44336;">
                                            R$ {business_metrics["prejuizo_fraudes"]:,.2f}
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Prejuízo de Fraudes
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            # 8) Métricas Operacionais
                            st.markdown("### 📊 Métricas Operacionais")
                            col_o1, col_o2, col_o3, col_o4 = st.columns(4, gap="large")

                            with col_o1:
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                            {business_metrics["num_legitimas_corretas"]:,}
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Legítimas Detectadas
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with col_o2:
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                            {business_metrics["num_fraudes_perdidas"]:,}
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Fraudes Perdidas
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with col_o3:
                                detection_rate = (
                                    (y_true.sum() - business_metrics["num_fraudes_perdidas"])
                                    / y_true.sum()
                                    * 100
                                ) if y_true.sum() > 0 else 0
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                            {detection_rate:.1f}%  
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Taxa de Detecção
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with col_o4:
                                fraud_pred_count = y_pred.sum()
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                            {fraud_pred_count:,}  
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Alertas de Fraude
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            # Caso não haja labels, mostra apenas contagem de alertas
                            st.markdown("### 🔍 Resultados da Predição (sem labels)")
                            fraud_pred_count = y_pred.sum()
                            fraud_pred_rate = (fraud_pred_count / len(y_pred)) * 100

                            col_p1, col_p2 = st.columns(2, gap="large")
                            with col_p1:
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                            {fraud_pred_count:,}  
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Alertas de Fraude
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with col_p2:
                                st.markdown(
                                    f"""
                                    <div style="background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                                                border-left: 4px solid #009ee3; 
                                                margin-bottom: 1rem;">
                                        <div style="font-size: 2rem; font-weight: bold; color: #1e88e5;">
                                            {fraud_pred_rate:.2f}%  
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                                            Taxa de Alerta
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        # ---------------------------------------------
                        # 9) Visualizações: Histograma de scores e Matriz
                        # ---------------------------------------------
                        st.markdown("### 📈 Visualizações")

                        # Histograma de Scores de Fraude
                        fig_scores = px.histogram(
                            x=y_pred_proba,
                            nbins=50,
                            title="Distribuição dos Scores de Fraude",
                            labels={"x": "Score de Fraude", "y": "Número de Transações"},
                            color_discrete_sequence=["#009ee3"],
                        )
                        fig_scores.add_vline(
                            x=DEFAULT_THRESHOLD,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Threshold: {DEFAULT_THRESHOLD}",
                            annotation_position="top right",
                        )
                        fig_scores.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)

                        # Matriz de Confusão (se houver labels)
                        if y_true is not None:
                            from sklearn.metrics import confusion_matrix

                            cm = confusion_matrix(y_true, y_pred)
                            fig_cm = px.imshow(
                                cm,
                                text_auto=True,
                                aspect="auto",
                                title="Matriz de Confusão",
                                labels=dict(x="Predição", y="Real"),
                                x=["Legítima", "Fraude"],
                                y=["Legítima", "Fraude"],
                                color_continuous_scale="Blues",
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)

                        # ---------------------------------------------
                        # 10) Transações Detectadas como Fraude
                        # ---------------------------------------------
                        st.markdown("### 🚨 Transações Detectadas como Fraude")
                        
                        # Cria DataFrame com resultados
                        df_results = df_refined.copy()
                        df_results["fraud_score"] = y_pred_proba
                        df_results["fraud_prediction"] = y_pred
                        
                        # Adiciona transaction_id se existir no dataset original
                        if "transaction_id" in df_raw.columns:
                            df_results["transaction_id"] = df_raw["transaction_id"].values
                        
                        # Adiciona is_fraud se existir
                        if y_true is not None:
                            df_results["is_fraud_real"] = y_true
                        
                        # Filtra apenas transações detectadas como fraude
                        fraud_transactions = df_results[df_results["fraud_prediction"] == 1].copy()
                        
                        if len(fraud_transactions) > 0:
                            # Ordena por score decrescente
                            fraud_transactions = fraud_transactions.sort_values("fraud_score", ascending=False)
                            
                            # Seleciona apenas as 4 colunas pedidas
                            display_cols = []
                            if "transaction_id" in fraud_transactions.columns:
                                display_cols.append("transaction_id")
                            display_cols.extend(["fraud_score", "fraud_prediction", "tx_amount"])
                            
                            st.markdown(f"**{len(fraud_transactions):,} transações detectadas como fraude:**")
                            st.dataframe(
                                fraud_transactions[display_cols],
                                use_container_width=True,
                                height=400
                            )
                            
                            # Se há labels reais, mostra estatísticas de acerto
                            if y_true is not None:
                                true_positives = (fraud_transactions["is_fraud_real"] == 1).sum()
                                false_positives = (fraud_transactions["is_fraud_real"] == 0).sum()
                                
                                col_acc1, col_acc2 = st.columns(2)
                                with col_acc1:
                                    st.metric("✅ Fraudes Reais Detectadas", true_positives)
                                with col_acc2:
                                    st.metric("❌ Falsos Positivos", false_positives)
                        else:
                            st.info("Nenhuma transação foi detectada como fraude com o threshold atual.")

                        # ---------------------------------------------
                        # 11) Download CSV com todas as predições
                        # ---------------------------------------------
                        st.markdown("### 📥 Download dos Resultados")
                        
                        # Prepara DataFrame para export (todas as transações)
                        export_data = []
                        if "transaction_id" in df_raw.columns:
                            export_data.append(df_raw["transaction_id"].values)
                            export_cols = ["transaction_id", "fraud_prediction"]
                        else:
                            export_cols = ["fraud_prediction"]
                        
                        export_data.append(y_pred)
                        
                        if len(export_data) > 1:
                            df_export = pd.DataFrame({
                                "transaction_id": export_data[0],
                                "fraud_prediction": export_data[1]
                            })
                        else:
                            df_export = pd.DataFrame({
                                "fraud_prediction": export_data[0]
                            })
                        
                        # Converte para CSV
                        csv_data = df_export.to_csv(index=False)
                        
                        st.download_button(
                            label="📊 Baixar Todas as Predições (CSV)",
                            data=csv_data,
                            file_name="fraud_predictions.csv",
                            mime="text/csv",
                            help=f"Download CSV com {len(df_export):,} transações e suas predições"
                        )

                    except Exception as e:
                        st.error(f"❌ Erro durante análise: {e}")
                        st.error("Verifique se as colunas do dataset batem com as do modelo.")

        except Exception as e:
            st.error(f"❌ Erro ao carregar dataset: {e}")
    else:
        # Se faltarem modelo ou dataset, exibe instruções
        st.markdown(
            """
            ### 🏁 Como usar:
            1. **Faça upload do dataset** (.feather) à esquerda  
            2. Aguarde que o modelo seja carregado automaticamente via MLflow (run ID no topo).  
            3. Clique em "🚀 Executar Análise de Fraudes" quando o modelo e dataset estiverem prontos.

            ### 📌 Detalhes importantes:
            - O modelo já está pré-configurado para rodar em `runs:/{RUN_ID}/pipeline-final`  
            - O Threshold padrão é 0.50 (caso queira alterar, ajuste a variável `DEFAULT_THRESHOLD` no topo)  
            - O Tracking URI usado é `MLFLOW_TRACKING_URI` (ajuste no topo, se necessário)  
            """
        )

    # ------------------------------------------------------------
    # Quadro de STATUS resumido
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Status Atual")
    if model_loaded:
        st.success("✅ Modelo MLflow carregado com sucesso")
    else:
        st.error("❌ Falha ao carregar o modelo MLflow")

    if uploaded_file:
        st.success("✅ Dataset carregado")
        df_info = pd.read_feather(uploaded_file)
        st.markdown(
            f"""
            **📊 Info Rápida do Dataset:**
            - {len(df_info):,} transações  
            - {len(df_info.columns)} colunas  
            - {'✅' if 'is_fraud' in df_info.columns else '❌'} Coluna `is_fraud`
            """
        )
        if st.button("👀 Ver Preview"):
            st.dataframe(df_info.head(3), use_container_width=True)
    else:
        st.info("⏳ Aguardando upload do dataset")

# =========================
# 8) FOOTER (rodapé)
# =========================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        💳 MercadoPago – Análise de Fraudes com MLflow
    </div>
    """,
    unsafe_allow_html=True,
)