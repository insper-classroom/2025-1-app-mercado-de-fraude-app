import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="MercadoPago - Detecção de Fraudes",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS  MercadoPago
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    
    .mp-header {
        background: linear-gradient(90deg, #009ee3 0%, #0073e6 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .mp-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    
    .mp-subtitle {
        color: #e3f2fd;
        font-size: 1.2rem;
        text-align: center;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #009ee3;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1e88e5;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .profit-positive {
        color: #4caf50;
    }
    
    .profit-negative {
        color: #f44336;
    }
    
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
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="mp-header">
    <h1 class="mp-title">💳 MercadoPago</h1>
    <p class="mp-subtitle">Sistema de Detecção de Fraudes</p>
</div>
""", unsafe_allow_html=True)

# Função de métrica de negócio
def business_profit_metric(y_true, y_pred, amount):
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
        'lucro_total': lucro_total,
        'lucro_legitimas': lucro_legitimas,
        'prejuizo_fraudes': prejuizo_fraudes,
        'num_legitimas_corretas': mask_true_negative.sum(),
        'num_fraudes_perdidas': mask_false_negative.sum()
    }

def refinar_dataset(df):
    import numpy as np

    # Remove colunas com None
    colunas_para_remover = ['model_score', 'tx_approve']
    df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns])

    # Conversões de data
    df['tx_datetime'] = pd.to_datetime(df['tx_datetime'])
    df['card_first_transaction'] = pd.to_datetime(df['card_first_transaction'])
    df['terminal_operation_start'] = pd.to_datetime(df['terminal_operation_start'])

    # Novas features temporais
    df['tx_hour'] = df['tx_datetime'].dt.hour
    df['tx_dow'] = df['tx_datetime'].dt.dayofweek
    df['card_age_days'] = (df['tx_datetime'] - df['card_first_transaction']).dt.days
    df['terminal_age_days'] = (df['tx_datetime'] - df['terminal_operation_start']).dt.days

    # Histórico da última transação
    df = df.sort_values(['card_id', 'tx_datetime'])
    df['tx_amount_prev'] = df.groupby('card_id')['tx_amount'].shift(1).fillna(0)
    df['hours_since_prev'] = (
        df['tx_datetime'] - df.groupby('card_id')['tx_datetime'].shift(1)
    ).dt.total_seconds().div(3600).fillna(0)

    # Cálculo de velocidade
    df['prev_lat'] = df.groupby('card_id')['latitude'].shift(1)
    df['prev_lon'] = df.groupby('card_id')['longitude'].shift(1)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    df['distance_km'] = haversine(df['prev_lat'], df['prev_lon'], df['latitude'], df['longitude'])
    df['travel_time_h'] = df['hours_since_prev'].replace(0, np.nan)
    df['speed_kmh'] = df['distance_km'] / df['travel_time_h']
    df['travel_speed'] = df['speed_kmh']

    # Remove colunas auxiliares
    df = df.drop(columns=['prev_lat', 'prev_lon', 'distance_km', 'travel_time_h', 'speed_kmh'], errors='ignore')

    return df

@st.cache_data
def convert_df_to_feather_bytes(df):
    import io
    buffer = io.BytesIO()
    df.reset_index(drop=True).to_feather(buffer)
    buffer.seek(0)
    return buffer


# Carregar modelo do MLflow
@st.cache_resource
def load_mlflow_model(model_uri):
    """Carrega modelo do MLflow"""
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model, True
    except Exception as e:
        return None, False

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configurações")
    
    # Input do modelo MLflow
    st.markdown("#### 🤖 Modelo MLflow")
    model_uri = st.text_input(
        "URI do Modelo",
        placeholder="runs:/run_id/model ou models:/model_name/version",
        help="Exemplo: runs://1234567890abcdef/model"
    )
    
    model_loaded = False
    model = None
    
    if model_uri:
        with st.spinner("Carregando modelo..."):
            model, model_loaded = load_mlflow_model(model_uri)
        
        if model_loaded:
            st.success("✅ Modelo carregado")
        else:
            st.error("❌ Erro ao carregar modelo")
    
    st.markdown("---")
    
    # Upload do dataset
    st.markdown("#### 📊 Dataset para Análise")
    uploaded_file = st.file_uploader(
        "📁 Carregue o dataset (.feather)",
        type=['feather'],
        help="Dataset com transações para análise"
    )
    
    if uploaded_file:
        try:
            df_preview = pd.read_feather(uploaded_file)
            df_preview = df_preview.dropna(axis=1, how='all')

            try:
                df_preview = refinar_dataset(df_preview)
            except Exception as e:
                st.warning("⚠️ Dataset bruto carregado. Refinamento não aplicado (colunas ausentes).")
            st.success(f"✅ {len(df_preview):,} transações")
            
            # Verificar se tem labels
            has_labels = 'is_fraud' in df_preview.columns
            if has_labels:
                fraud_count = df_preview['is_fraud'].sum()
                st.info(f"🎯 {fraud_count:,} fraudes reais")
            else:
                st.warning("⚠️ Sem labels (is_fraud)")
                
        except Exception as e:
            st.error("❌ Erro ao ler dataset")
    
    st.markdown("---")
    
    # Configuração de threshold
    st.markdown("#### ⚙️ Configuração")
    threshold = st.slider(
        "🎯 Threshold de Fraude",
        0.0, 1.0, 0.5, 0.01,
        help="Score mínimo para classificar como fraude"
    )
    
    st.caption(f"Threshold: {threshold}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_file and model_loaded:
        try:
            # Carregar dataset
            df = pd.read_feather(uploaded_file)
            df_preview = df_preview.dropna(axis=1, how='all')
            df = refinar_dataset(df)

            # Gera botão de download do dataset refinado
            feather_buffer = convert_df_to_feather_bytes(df)
            st.download_button(
                label="📥 Baixar Dataset Refinado",
                data=feather_buffer,
                file_name="dataset_refinado.feather",
                mime="application/octet-stream"
            )
            
            st.markdown("### 📋 Informações do Dataset")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-label">Total de Transações</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                if 'is_fraud' in df.columns:
                    fraud_count = df['is_fraud'].sum()
                    fraud_rate = (fraud_count / len(df)) * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{fraud_count:,}</div>
                        <div class="metric-label">Fraudes Reais ({fraud_rate:.2f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">N/A</div>
                        <div class="metric-label">Fraudes (sem labels)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_info3:
                if 'tx_amount' in df.columns:
                    total_amount = df['tx_amount'].sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">R$ {total_amount:,.2f}</div>
                        <div class="metric-label">Volume Total</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Botão de análise
            if st.button("🚀 Executar Análise de Fraudes", key="analyze"):
                with st.spinner("Fazendo predições..."):
                    try:
                        # Preparar dados para predição
                        if 'is_fraud' in df.columns:
                            X_test = df.drop(columns=['is_fraud'])
                            y_true = df['is_fraud'].values
                            has_labels = True
                        else:
                            X_test = df.copy()
                            y_true = None
                            has_labels = False
                        
                        # Fazer predições com modelo MLflow
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                        else:
                            # Se não tem predict_proba, usar predict
                            y_pred_binary = model.predict(X_test)
                            y_pred_proba = y_pred_binary.astype(float)
                        
                        y_pred = (y_pred_proba >= threshold).astype(int)
                        
                        # Calcular métricas de negócio
                        amounts = X_test['tx_amount'].values if 'tx_amount' in X_test.columns else np.ones(len(X_test))
                        
                        if has_labels:
                            business_metrics = business_profit_metric(y_true, y_pred, amounts)
                            
                            st.markdown("### 💰 Resultados Financeiros")
                            
                            col_res1, col_res2, col_res3 = st.columns(3)
                            
                            with col_res1:
                                profit_class = "profit-positive" if business_metrics['lucro_total'] >= 0 else "profit-negative"
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value {profit_class}">R$ {business_metrics['lucro_total']:,.2f}</div>
                                    <div class="metric-label">Lucro/Prejuízo Total</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_res2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value profit-positive">R$ {business_metrics['lucro_legitimas']:,.2f}</div>
                                    <div class="metric-label">Lucro de Legítimas</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_res3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value profit-negative">R$ {business_metrics['prejuizo_fraudes']:,.2f}</div>
                                    <div class="metric-label">Prejuízo de Fraudes</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Métricas operacionais
                            st.markdown("### 📊 Métricas Operacionais")
                            
                            col_op1, col_op2, col_op3, col_op4 = st.columns(4)
                            
                            with col_op1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{business_metrics['num_legitimas_corretas']:,}</div>
                                    <div class="metric-label">Legítimas Detectadas</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_op2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{business_metrics['num_fraudes_perdidas']:,}</div>
                                    <div class="metric-label">Fraudes Perdidas</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_op3:
                                detection_rate = ((y_true.sum() - business_metrics['num_fraudes_perdidas']) / y_true.sum() * 100) if y_true.sum() > 0 else 0
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{detection_rate:.1f}%</div>
                                    <div class="metric-label">Taxa de Detecção</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_op4:
                                fraud_pred_count = y_pred.sum()
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{fraud_pred_count:,}</div>
                                    <div class="metric-label">Alertas de Fraude</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        else:
                            # Sem labels, apenas mostrar predições
                            st.markdown("### 🔍 Resultados da Predição")
                            
                            fraud_pred_count = y_pred.sum()
                            fraud_pred_rate = (fraud_pred_count / len(y_pred)) * 100
                            
                            col_pred1, col_pred2 = st.columns(2)
                            
                            with col_pred1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{fraud_pred_count:,}</div>
                                    <div class="metric-label">Alertas de Fraude</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_pred2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{fraud_pred_rate:.2f}%</div>
                                    <div class="metric-label">Taxa de Alerta</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Visualizações
                        st.markdown("### 📈 Visualizações")
                        
                        # Distribuição de scores
                        fig_scores = px.histogram(
                            x=y_pred_proba, 
                            nbins=50,
                            title="Distribuição dos Scores de Fraude",
                            labels={'x': 'Score de Fraude', 'y': 'Número de Transações'},
                            color_discrete_sequence=['#009ee3']
                        )
                        fig_scores.add_vline(
                            x=threshold, 
                            line_dash="dash", 
                            line_color="red", 
                            annotation_text=f"Threshold: {threshold}"
                        )
                        fig_scores.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)
                        
                        if has_labels:
                            # Matriz de confusão
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(y_true, y_pred)
                            
                            fig_cm = px.imshow(
                                cm,
                                text_auto=True,
                                aspect="auto",
                                title="Matriz de Confusão",
                                labels=dict(x="Predição", y="Real"),
                                x=['Legítima', 'Fraude'],
                                y=['Legítima', 'Fraude'],
                                color_continuous_scale='Blues'
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # Top transações suspeitas
                        st.markdown("### 🔍 Top 10 Transações Mais Suspeitas")
                        
                        df_results = df.copy()
                        df_results['fraud_score'] = y_pred_proba
                        df_results['fraud_prediction'] = y_pred
                        
                        top_fraud_scores = df_results.nlargest(10, 'fraud_score')
                        
                        # Selecionar colunas relevantes para mostrar
                        display_cols = ['fraud_score', 'fraud_prediction']
                        if 'tx_amount' in top_fraud_scores.columns:
                            display_cols.insert(0, 'tx_amount')
                        if 'is_fraud' in top_fraud_scores.columns:
                            display_cols.append('is_fraud')
                        
                        # Adicionar outras colunas interessantes se existirem
                        for col in ['tx_hour', 'estado', 'travel_speed']:
                            if col in top_fraud_scores.columns:
                                display_cols.append(col)
                        
                        st.dataframe(
                            top_fraud_scores[display_cols],
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante análise: {e}")
                        st.error("Verifique se o modelo é compatível com o dataset.")
        
        except Exception as e:
            st.error(f"❌ Erro ao carregar dataset: {e}")
    
    else:
        # Instruções
        st.markdown("""
        ### 🏁 Como usar:
        
        1. **🤖 Insira a URI do seu modelo MLflow** na barra lateral
        2. **📁 Faça upload do dataset** (.feather) 
        3. **⚙️ Configure o threshold** (opcional)
        4. **🚀 Execute a análise** para ver lucro/prejuízo
        
        ### 💰 Métricas Calculadas:
        
        - **💚 Lucro**: 3% das transações legítimas aprovadas
        - **💔 Prejuízo**: 100% das fraudes não detectadas
        - **📊 ROI Total**: Lucro - Prejuízo
        
        ### 🔗 Exemplo de URI MLflow:
        
        ```
        runs://1234567890abcdef/model
        models://fraud_model/Production
        models://fraud_model/1
        ```
        """)

with col2:
    st.markdown("### 📋 Status")
    
    # Status do modelo
    if model_uri and model_loaded:
        st.success("✅ Modelo MLflow")
    elif model_uri:
        st.error("❌ Erro no modelo")
    else:
        st.info("⏳ Aguardando URI")
    
    # Status do dataset
    if uploaded_file:
        st.success("✅ Dataset carregado")
        
        # Info rápida
        df_info = pd.read_feather(uploaded_file)
        st.markdown(f"""
        **📊 Info:**
        - {len(df_info):,} transações
        - {len(df_info.columns)} colunas
        - {'✅' if 'is_fraud' in df_info.columns else '❌'} Labels
        """)
        
        if st.button("👀 Preview"):
            st.dataframe(df_info.head(3), use_container_width=True)
    else:
        st.info("⏳ Aguardando dataset")
    
    st.markdown("---")
    
    st.markdown("""
    ### 💡 Dicas:
    
    **🎯 Threshold:**
    - Baixo = + alertas
    - Alto = + conservador
    
    **📊 Dataset:**
    - Precisa de `tx_amount`
    - `is_fraud` opcional
    
    **🤖 Modelo:**
    - Deve ter `predict_proba`
    - Compatível com colunas
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    💳 MercadoPago - Análise de Fraudes com MLflow
</div>
""", unsafe_allow_html=True)