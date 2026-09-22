import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# ─── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="LocaPredict – AIOps Dashboard",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paleta de cores ─────────────────────────────────────────────────────────
CORES = {
    "vermelho":    "#E8003D",   # violação / negativo
    "ciano":       "#00C1D4",   # info / positivo leve
    "azul_escuro": "#0A0E1A",   # fundo base
    "branco":      "#E8ECF4",   # texto principal
    "cinza":       "#5E6680",   # texto secundário
    "alerta":      "#FF6B35",   # atenção alta
    "sucesso":     "#00B96B",   # cumprido / positivo
    "ambar":       "#FFA02F",   # âmbar
}

ORDEM_PRIORIDADE = ["1 - Crítica", "2 - Alta", "3 - Média", "4 - Baixa", "5 - Muito Baixa"]
CORES_PRIORIDADE = {
    "1 - Crítica":     CORES["vermelho"],
    "2 - Alta":        CORES["alerta"],
    "3 - Média":       CORES["ambar"],
    "4 - Baixa":       CORES["ciano"],
    "5 - Muito Baixa": CORES["cinza"],
}

# ─── CSS  ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    /* fundo e sidebar no navy-preto do terminal */
    [data-testid="stAppViewContainer"] {
        background: #0A0E1A;
        font-family: 'IBM Plex Mono', monospace;
    }
    [data-testid="stSidebar"] {
        background: #0D1220;
        border-right: 1px solid #2A3050;
    }

    /* cartões de métrica: flat, sem gradiente, borda hairline */
    .cartao-metrica {
        background: #11172A;
        border: 1px solid #2A3050;
        border-radius: 0;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .cartao-metrica .valor {
        font-size: 2rem;
        font-weight: 600;
        color: #FFA02F;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }
    .cartao-metrica .rotulo {
        font-size: 0.68rem;
        color: #5E6680;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .cartao-metrica .delta {
        font-size: 0.7rem;
        margin-top: 4px;
        color: #5E6680;
    }

    /* títulos de seção — traço vertical âmbar + uppercase monospace */
    .titulo-secao {
        font-size: 0.75rem;
        font-weight: 600;
        color: #FFA02F;
        font-family: 'IBM Plex Mono', monospace;
        border-left: 2px solid #FFA02F;
        padding-left: 10px;
        margin: 24px 0 14px 0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* caixas de alerta e info — borda sólida lateral, sem radius */
    .caixa-alerta {
        background: rgba(232, 0, 61, 0.06);
        border: 1px solid #2A3050;
        border-left: 3px solid #E8003D;
        border-radius: 0;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #E8ECF4;
    }
    .caixa-info {
        background: rgba(0, 193, 212, 0.05);
        border: 1px solid #2A3050;
        border-left: 3px solid #00C1D4;
        border-radius: 0;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #E8ECF4;
    }

    /* textos gerais */
    h1, h2, h3, h4, p, li {
        color: #E8ECF4 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    /* abas — uppercase monospace, underline âmbar na selecionada */
    .stTabs [data-baseweb="tab"] {
        color: #5E6680;
        font-weight: 500;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        color: #FFA02F !important;
        border-bottom: 2px solid #FFA02F;
    }

    /* labels dos controles da sidebar */
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stDateInput label {
        color: #5E6680 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* divisores */
    hr { border-color: #2A3050 !important; }

    /* dataframes — tema escuro alinhado */
    [data-testid="stDataFrame"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Carregamento dos dados ──────────────────────────────────────────────────
# usando cache pra não reler o Excel toda vez que algum filtro muda
@st.cache_data(show_spinner="Carregando dataset…")
def carregar_dados():
    # ancoro no diretório do próprio script pra não depender de onde o terminal foi aberto
    pasta_base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_excel(os.path.join(pasta_base, "LW-DATASET.xlsx"))

    # datas — Resolvido pode ter nulos, então errors='coerce' em vez de explodir
    df["Aberto"]   = pd.to_datetime(df["Aberto"])
    df["Resolvido"] = pd.to_datetime(df["Resolvido"], errors="coerce")

    # colunas auxiliares pra facilitar os filtros e agrupamentos
    df["data"]       = df["Aberto"].dt.date
    df["hora"]       = df["Aberto"].dt.hour
    df["dia_semana"] = df["Aberto"].dt.day_name()
    df["mes"]        = df["Aberto"].dt.month
    df["ano"]        = df["Aberto"].dt.year
    df["semana"]     = df["Aberto"].dt.isocalendar().week.astype(int)

    # extraio o número da prioridade ("2 - Alta" → 2) pra usar como feature no clustering
    df["prioridade_num"] = df["Prioridade"].str[0].astype(int)

    # binário pra somar e calcular médias facilmente depois
    df["kpi_violado_bin"] = (df["KPI Violado?"] == "SIM").astype(int)
    df["entrou_kpi_bin"]  = (df["Entrou para KPI?"] == "SIM").astype(int)

    return df

df_completo = carregar_dados()


# ─── Sidebar / filtros ───────────────────────────────────────────────────────
with st.sidebar:
    # cabeçalho estilo painel de controle — sem emoji, texto estruturado
    st.markdown("""
    <div style='padding: 16px 0 20px; border-bottom: 1px solid #2A3050; margin-bottom: 16px;'>
        <div style='font-family: "IBM Plex Mono", monospace; font-size: 0.6rem;
                    color: #5E6680; text-transform: uppercase; letter-spacing: 0.12em;
                    margin-bottom: 6px;'>SISTEMA</div>
        <div style='font-family: "IBM Plex Mono", monospace; font-size: 1.15rem;
                    font-weight: 600; color: #E8ECF4; letter-spacing: -0.01em;'>
            LOCAPREDICT
        </div>
        <div style='font-family: "IBM Plex Mono", monospace; font-size: 0.68rem;
                    color: #FFA02F; margin-top: 4px; letter-spacing: 0.04em;'>
            AIOps · Locaweb + FIAP 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    # trabalho só com 2025 porque é o ano com dados mais completos
    df_2025     = df_completo[df_completo["ano"] == 2025].copy()
    data_minima = df_2025["data"].min()
    data_maxima = df_2025["data"].max()

    st.markdown("<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.65rem;color:#FFA02F;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>▸ PERÍODO</div>", unsafe_allow_html=True)
    intervalo_datas = st.date_input(
        "Selecione o intervalo",
        value=(pd.to_datetime("2025-01-01").date(), data_maxima),
        min_value=data_minima, max_value=data_maxima,
    )

    st.markdown("<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.65rem;color:#FFA02F;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;margin-top:16px;'>▸ PRIORIDADE</div>", unsafe_allow_html=True)
    prioridades_selecionadas = st.multiselect(
        "Filtrar por prioridade",
        options=ORDEM_PRIORIDADE,
        default=["2 - Alta", "3 - Média"],
    )
    # se o usuário deselecionar tudo, mostra geral — evita tela em branco
    if not prioridades_selecionadas:
        prioridades_selecionadas = ORDEM_PRIORIDADE

    st.markdown("<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.65rem;color:#FFA02F;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;margin-top:16px;'>▸ PRODUTO (TOP 15)</div>", unsafe_allow_html=True)
    produtos_top          = df_2025["Produto"].value_counts().head(15).index.tolist()
    produtos_selecionados = st.multiselect("Filtrar produtos", options=produtos_top, default=[])

    st.markdown("---")
    st.markdown(
        "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.6rem;"
        "color:#2A3050;text-align:center;'>Attack On Data · 2TSCOA</div>",
        unsafe_allow_html=True
    )


# ─── Aplicação dos filtros ───────────────────────────────────────────────────
# se o usuário selecionou o intervalo completo (2 datas), uso elas; senão pego o range todo
if len(intervalo_datas) == 2:
    data_inicio, data_fim = intervalo_datas
else:
    data_inicio, data_fim = df_2025["data"].min(), df_2025["data"].max()

# filtro principal por data e produto
df = df_2025[(df_2025["data"] >= data_inicio) & (df_2025["data"] <= data_fim)]
if produtos_selecionados:
    df = df[df["Produto"].isin(produtos_selecionados)]

# subconjunto por prioridade — disponível pra uso em abas específicas se precisar
df_prioridade = df[df["Prioridade"].isin(prioridades_selecionadas)]


# ─── Cabeçalho principal ─────────────────────────────────────────────────────
_, coluna_titulo = st.columns([1, 6])
with coluna_titulo:
    st.markdown("""
    <div style='margin-bottom: 2px;'>
        <span style='font-family: "IBM Plex Mono", monospace; font-size: 0.6rem;
                     color: #5E6680; text-transform: uppercase; letter-spacing: 0.12em;'>
            MONITOR OPERACIONAL
        </span>
    </div>
    <h1 style='margin: 0; font-size: 1.65rem; font-weight: 700;
               font-family: "IBM Plex Sans", sans-serif;
               color: #E8ECF4 !important; letter-spacing: -0.01em;'>
        LocaPredict — <span style='color:#E8003D;'>AIOps</span> Dashboard
    </h1>
    <p style='color: #5E6680; margin-top: 4px; font-size: 0.72rem;
              font-family: "IBM Plex Mono", monospace; letter-spacing: 0.02em;'>
        Previsão de Incidentes & Tendências Operacionais · Locaweb + FIAP Challenge 2026
    </p>
    """, unsafe_allow_html=True)


# ─── Abas principais ─────────────────────────────────────────────────────────
aba1, aba2, aba3, aba4 = st.tabs([
    "VISÃO GERAL",
    "FORECASTING  D+1 / D+7",
    "CLUSTERIZAÇÃO",
    "RISCO DE OLA",
])

# configuração de layout padrão pros gráficos — reutilizo em todos pra manter consistência
_layout_base = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#5E6680", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=10, b=10),
    xaxis=dict(gridcolor="#1A2138", linecolor="#2A3050"),
    yaxis=dict(gridcolor="#1A2138", linecolor="#2A3050"),
)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 – VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
with aba1:

    # ── Métricas principais ──────────────────────────────────────────────────
    total_incidentes = len(df)
    kpi_entradas     = df["entrou_kpi_bin"].sum()
    kpi_violados     = df["kpi_violado_bin"].sum()
    pct_violados     = round(kpi_violados / kpi_entradas * 100, 1) if kpi_entradas > 0 else 0
    duracao_media    = round(df["Duração"].mean() / 60, 1)  # segundos → minutos
    contagem_p2      = len(df[df["Prioridade"] == "2 - Alta"])

    col1, col2, col3, col4, col5 = st.columns(5)
    cartoes = [
        (col1, total_incidentes,    "TOTAL INCIDENTES",   CORES["ciano"]),
        (col2, f"{pct_violados}%",  "OLA VIOLADO",        CORES["vermelho"]),
        (col3, kpi_entradas,        "ENTROU NO KPI",      CORES["ambar"]),
        (col4, contagem_p2,         "ALTA PRIORIDADE P2", CORES["alerta"]),
        (col5, f"{duracao_media}m", "DURAÇÃO MÉDIA",      CORES["sucesso"]),
    ]
    for coluna, valor, rotulo, cor in cartoes:
        with coluna:
            st.markdown(f"""
            <div class='cartao-metrica'>
                <div class='valor' style='color:{cor}'>{f"{valor:,}" if isinstance(valor, int) else valor}</div>
                <div class='rotulo'>{rotulo}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tendência diária ─────────────────────────────────────────────────────
    st.markdown("<div class='titulo-secao'>▸ Tendência Diária de Incidentes</div>", unsafe_allow_html=True)

    diario = df.groupby("data").size().reset_index(name="total")
    diario["data"]          = pd.to_datetime(diario["data"])
    diario["media_movel_7d"] = diario["total"].rolling(7, min_periods=1).mean()

    grafico_tendencia = go.Figure()
    grafico_tendencia.add_trace(go.Bar(
        x=diario["data"], y=diario["total"],
        name="Incidentes/dia", marker_color=CORES["ciano"], opacity=0.35
    ))
    grafico_tendencia.add_trace(go.Scatter(
        x=diario["data"], y=diario["media_movel_7d"],
        name="Média Móvel 7d", line=dict(color=CORES["vermelho"], width=2)
    ))
    grafico_tendencia.update_layout(**{**_layout_base, "height": 280})
    st.plotly_chart(grafico_tendencia, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='titulo-secao'>▸ Distribuição por Prioridade</div>", unsafe_allow_html=True)
        contagem_prioridades = df["Prioridade"].value_counts().reindex(ORDEM_PRIORIDADE).dropna()
        grafico_prioridade = px.bar(
            contagem_prioridades.reset_index(), x="Prioridade", y="count",
            color="Prioridade",
            color_discrete_map=CORES_PRIORIDADE,
            labels={"count": "Incidentes"},
        )
        grafico_prioridade.update_layout(**{**_layout_base, "height": 280, "showlegend": False})
        st.plotly_chart(grafico_prioridade, use_container_width=True)

    with col_b:
        st.markdown("<div class='titulo-secao'>▸ Heatmap: Hora × Dia da Semana</div>", unsafe_allow_html=True)

        # ordeno os dias em inglês e substituo o índice pra exibir em PT depois
        ordem_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dias_pt = {
            "Monday":"Seg", "Tuesday":"Ter", "Wednesday":"Qua",
            "Thursday":"Qui", "Friday":"Sex", "Saturday":"Sáb", "Sunday":"Dom"
        }
        mapa_calor = df.groupby(["dia_semana", "hora"]).size().reset_index(name="count")
        mapa_calor_pivot = mapa_calor.pivot(
            index="dia_semana", columns="hora", values="count"
        ).reindex(ordem_dias)
        mapa_calor_pivot.index = [dias_pt[d] for d in mapa_calor_pivot.index]

        grafico_heatmap = px.imshow(
            mapa_calor_pivot,
            color_continuous_scale=[[0, "#0A0E1A"], [0.5, "#00C1D4"], [1, "#E8003D"]],
            labels={"x": "Hora do Dia", "y": "Dia", "color": "Qtd"},
            aspect="auto",
        )
        grafico_heatmap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color="#5E6680", size=10),
            margin=dict(t=10, b=10), height=280,
            coloraxis_showscale=False,
        )
        st.plotly_chart(grafico_heatmap, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("<div class='titulo-secao'>▸ Top 10 Produtos com Mais Incidentes</div>", unsafe_allow_html=True)
        top10_produtos = df["Produto"].value_counts().head(10).reset_index()
        top10_produtos.columns = ["Produto", "Incidentes"]
        grafico_produtos = px.bar(
            top10_produtos.sort_values("Incidentes"), x="Incidentes", y="Produto",
            orientation="h", color="Incidentes",
            color_continuous_scale=["#1A2138", "#00C1D4"],
        )
        grafico_produtos.update_layout(
            **{**_layout_base, "height": 320, "coloraxis_showscale": False}
        )
        st.plotly_chart(grafico_produtos, use_container_width=True)

    with col_d:
        st.markdown("<div class='titulo-secao'>▸ Top 10 Categorias</div>", unsafe_allow_html=True)
        top10_categorias = df["Categoria"].value_counts().head(10).reset_index()
        top10_categorias.columns = ["Categoria", "Incidentes"]
        grafico_categorias = px.pie(
            top10_categorias, names="Categoria", values="Incidentes",
            color_discrete_sequence=[
                CORES["vermelho"], CORES["ciano"], CORES["alerta"],
                CORES["ambar"],    CORES["sucesso"], "#7B61FF",
                "#FF6EC7", "#44BBA4", "#E94F37", "#393E41"
            ],
            hole=0.45,
        )
        grafico_categorias.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color="#5E6680", size=10),
            margin=dict(t=10, b=10), height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        )
        st.plotly_chart(grafico_categorias, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 – FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
with aba2:
    st.markdown("""
    <div class='caixa-info'>
        <b>MODELO PREDITIVO</b> — Suavização Exponencial (Holt-Winters) com sazonalidade semanal.
        Previsão de volume de incidentes para <b>D+1</b> e <b>D+7</b>.
    </div>
    """, unsafe_allow_html=True)

    _, col_controles = st.columns([3, 1])
    with col_controles:
        previsao_prioridade = st.selectbox(
            "Prioridade para previsão",
            ["Todas", "2 - Alta", "3 - Média", "2 - Alta + 3 - Média"]
        )
        dias_historico = st.slider("Histórico para treino (dias)", 30, 180, 90, 10)

    # monto a série de acordo com o filtro de prioridade escolhido
    if previsao_prioridade == "Todas":
        df_previsao = df_2025.copy()
    elif previsao_prioridade == "2 - Alta + 3 - Média":
        df_previsao = df_2025[df_2025["Prioridade"].isin(["2 - Alta", "3 - Média"])]
    else:
        df_previsao = df_2025[df_2025["Prioridade"] == previsao_prioridade]

    # agrupo por dia e preencho os buracos da série com zero
    diario_previsao = df_previsao.groupby("data").size().reset_index(name="total")
    diario_previsao["data"] = pd.to_datetime(diario_previsao["data"])
    diario_previsao = diario_previsao.set_index("data").asfreq("D", fill_value=0)

    serie_treino = diario_previsao["total"].astype(float).tail(dias_historico)

    try:
        modelo = ExponentialSmoothing(
            serie_treino,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
        ajuste = modelo.fit(optimized=True)
        previsao_8 = ajuste.forecast(8)

        datas_previsao = previsao_8.index
        valores_prev = previsao_8.values
        largura_ic = serie_treino.std() * 1.5

        val_d1 = int(round(max(0, valores_prev[0])))
        val_d7 = int(round(max(0, valores_prev[6])))
        soma_7d = int(round(sum(max(0, v) for v in valores_prev[:7])))


        # cartões de previsão
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""<div class='cartao-metrica'>
                <div class='valor' style='color:{CORES["ambar"]}'>{val_d1}</div>
                <div class='rotulo'>PREVISÃO D+1</div>
                <div class='delta'>próximo dia</div>
            </div>""", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""<div class='cartao-metrica'>
                <div class='valor' style='color:{CORES["alerta"]}'>{val_d7}</div>
                <div class='rotulo'>PREVISÃO D+7</div>
                <div class='delta'>sétimo dia</div>
            </div>""", unsafe_allow_html=True)
        with col_m3:
                st.markdown(f"""<div class='cartao-metrica'>
                <div class='valor' style='color:{CORES["vermelho"]}'>{soma_7d}</div>
                <div class='rotulo'>TOTAL 7 DIAS</div>
                <div class='delta'>volume acumulado</div>
            </div>""", unsafe_allow_html=True)

            # gráfico com histórico + previsão + banda de confiança
        historico_plot = diario_previsao.tail(dias_historico)
        grafico_previsao = go.Figure()
        grafico_previsao.add_trace(go.Scatter(
            x=historico_plot.index, y=historico_plot["total"],
            name="Histórico", line=dict(color=CORES["ciano"], width=1.5),
            fill="tozeroy", fillcolor="rgba(0,193,212,0.06)"
        ))
        grafico_previsao.add_trace(go.Scatter(
            x=datas_previsao, y=[max(0, v) for v in valores_prev],
            name="Previsão",
            line=dict(color=CORES["ambar"], width=2, dash="dot"),
            mode="lines+markers", marker=dict(size=6, color=CORES["ambar"])
        ))
        # banda de confiança — faixa translúcida âmbar
        limite_superior = [max(0, v + largura_ic) for v in valores_prev]
        limite_inferior = [max(0, v - largura_ic) for v in valores_prev]
        grafico_previsao.add_trace(go.Scatter(
            x=list(datas_previsao) + list(datas_previsao[::-1]),
            y=limite_superior + limite_inferior[::-1],
            fill="toself", fillcolor="rgba(255,160,47,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="Intervalo de Confiança"
        ))

        # linha vertical marcando onde começa a previsão
        x_corte = datas_previsao[0]
        grafico_previsao.add_shape(
            type="line",
            x0=x_corte, x1=x_corte,
            y0=0, y1=1, yref="paper",
            line=dict(color=CORES["ciano"], width=1, dash="dash"),
        )
        grafico_previsao.add_annotation(
            x=x_corte, y=1, yref="paper",
            text="D+1", showarrow=False,
            xanchor="left", yanchor="bottom",
            font=dict(color=CORES["ciano"]),
        )
        grafico_previsao.update_layout(**{
            **_layout_base,
            "height": 350,
            "margin": dict(t=20, b=10),
            "title": dict(
                text=f"FORECASTING — {previsao_prioridade}",
                font=dict(color="#E8ECF4", family="IBM Plex Mono", size=11)
            )
        })
        st.plotly_chart(grafico_previsao, use_container_width=True)

        # tabela com os 8 dias previstos
        st.markdown("<div class='titulo-secao'>▸ Tabela de Previsão — Próximos 8 Dias</div>",
                    unsafe_allow_html=True)
        df_tabela_previsao = pd.DataFrame({
            "Dia": [f"D+{i + 1}" for i in range(8)],
            "Data": datas_previsao.strftime("%d/%m/%Y"),
            "Dia da Semana": datas_previsao.day_name(),
            "Previsão": [int(round(max(0, v))) for v in valores_prev],
            "Mín": [int(round(max(0, v - largura_ic))) for v in valores_prev],
            "Máx": [int(round(max(0, v + largura_ic))) for v in valores_prev],
        })
        st.dataframe(
            df_tabela_previsao.style.background_gradient(subset=["Previsão"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True,
        )

    except Exception as erro:
            st.error(f"Erro no modelo: {erro}. Tente aumentar o período de treino.")

    # ── Sazonalidade semanal ─────────────────────────────────────────────────
    st.markdown("<div class='titulo-secao'>▸ Padrão de Sazonalidade Semanal</div>", unsafe_allow_html=True)

    # uso df_previsao aqui pra respeitar o filtro de prioridade escolhido
    media_dia_semana = df_previsao.groupby(df_previsao["Aberto"].dt.day_of_week).size()
    media_dia_semana.index = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    grafico_sazonalidade = px.bar(
        media_dia_semana.reset_index(), x="index", y=0,
        labels={"index": "Dia", 0: "Incidentes"},
        color=0, color_continuous_scale=["#1A2138", "#E8003D"],
    )
    grafico_sazonalidade.update_layout(**{
        **_layout_base, "height": 220, "coloraxis_showscale": False
    })
    st.plotly_chart(grafico_sazonalidade, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 – CLUSTERIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
with aba3:
    st.markdown("""
    <div class='caixa-info'>
        <b>CLUSTERIZAÇÃO K-MEANS</b> — Agrupamento de incidentes por perfil operacional
        (prioridade, hora, dia da semana, duração) para identificar padrões ocultos e causas raiz.
    </div>
    """, unsafe_allow_html=True)

    _, col_controles_cluster = st.columns([3, 1])
    with col_controles_cluster:
        num_clusters    = st.slider("Número de clusters (K)", 2, 8, 4)
        tamanho_amostra = st.slider("Amostra (mil registros)", 5, 50, 20) * 1000

    @st.cache_data(show_spinner="Rodando K-Means…")
    def executar_clusterizacao(n_k, n_amostra, data_s, data_e):
        # filtro pela janela de data selecionada na sidebar
        df_cluster = df_2025[
            (df_2025["data"] >= data_s) & (df_2025["data"] <= data_e)
        ].copy()

        # se o dataset for grande demais, sorteo aleatório pra não travar
        df_cluster = df_cluster.sample(min(n_amostra, len(df_cluster)), random_state=42)

        # features que fazem sentido operacionalmente pra agrupar incidentes
        caracteristicas = pd.DataFrame({
            "prioridade":  df_cluster["prioridade_num"],
            "hora":        df_cluster["hora"],
            "dia_semana":  df_cluster["Aberto"].dt.day_of_week,
            "duracao_log": np.log1p(df_cluster["Duração"]),   # log pra reduzir impacto de outliers
            "entrou_kpi":  df_cluster["entrou_kpi_bin"],
            "kpi_violado": df_cluster["kpi_violado_bin"],
        })

        # normalizo antes do KMeans porque as escalas são muito diferentes
        normalizador = StandardScaler()
        X = normalizador.fit_transform(caracteristicas)

        # rodo o KMeans e atribuo os rótulos
        kmeans = KMeans(n_clusters=n_k, random_state=42, n_init="auto")
        rotulos = kmeans.fit_predict(X)

        # PCA pra 2D — só pra visualização, não muda o clustering em si
        pca  = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        df_cluster["cluster"] = rotulos
        df_cluster["pca_1"]   = X_2d[:, 0]
        df_cluster["pca_2"]   = X_2d[:, 1]

        # silhouette mede quão bem separados ficaram os clusters
        pontuacao = silhouette_score(X, rotulos, sample_size=5000, random_state=42)
        return df_cluster, pontuacao

    df_clusterizado, silhouette_val = executar_clusterizacao(
        num_clusters, tamanho_amostra, data_inicio, data_fim
    )

    # ── Silhouette score ─────────────────────────────────────────────────────
    col_silhouette, _, _ = st.columns(3)
    with col_silhouette:
        # verde se bom, amarelo se ok, vermelho se ruim
        cor_silhouette = (
            CORES["sucesso"] if silhouette_val > 0.4
            else CORES["ambar"] if silhouette_val > 0.25
            else CORES["vermelho"]
        )
        st.markdown(f"""<div class='cartao-metrica'>
            <div class='valor' style='color:{cor_silhouette}'>{silhouette_val:.3f}</div>
            <div class='rotulo'>SILHOUETTE SCORE — qualidade dos clusters</div>
        </div>""", unsafe_allow_html=True)

    col_cl_a, col_cl_b = st.columns(2)

    with col_cl_a:
        st.markdown("<div class='titulo-secao'>▸ Mapa PCA dos Clusters</div>", unsafe_allow_html=True)
        paleta_clusters = px.colors.qualitative.Bold
        # limito a 5000 pontos pra não travar o browser
        amostra_viz = df_clusterizado.sample(min(5000, len(df_clusterizado)), random_state=1)
        grafico_pca = px.scatter(
            amostra_viz,
            x="pca_1", y="pca_2",
            color=amostra_viz["cluster"].astype(str),
            color_discrete_sequence=paleta_clusters,
            labels={"color": "Cluster", "pca_1": "PC1", "pca_2": "PC2"},
            opacity=0.6,
        )
        grafico_pca.update_layout(**{**_layout_base, "height": 340})
        st.plotly_chart(grafico_pca, use_container_width=True)

    with col_cl_b:
        st.markdown("<div class='titulo-secao'>▸ Perfil dos Clusters</div>", unsafe_allow_html=True)
        perfil_clusters = df_clusterizado.groupby("cluster").agg(
            total           = ("cluster",        "count"),
            prio_media      = ("prioridade_num",  "mean"),
            hora_media      = ("hora",            "mean"),
            duracao_media   = ("Duração",         "mean"),
            pct_kpi_violado = ("kpi_violado_bin", "mean"),
        ).round(2).reset_index()
        perfil_clusters["pct_kpi_violado"] = (
            perfil_clusters["pct_kpi_violado"] * 100
        ).round(1).astype(str) + "%"
        perfil_clusters["cluster"] = "Cluster " + perfil_clusters["cluster"].astype(str)
        perfil_clusters.columns = [
            "Cluster", "Total", "Prioridade Média",
            "Hora Média", "Duração Média (s)", "% KPI Violado"
        ]
        st.dataframe(perfil_clusters, use_container_width=True, hide_index=True)

    # ── Prioridade por cluster ───────────────────────────────────────────────
    st.markdown("<div class='titulo-secao'>▸ Prioridade por Cluster</div>", unsafe_allow_html=True)
    contagem_cluster_prio = df_clusterizado.groupby(
        ["cluster", "Prioridade"]
    ).size().reset_index(name="count")
    contagem_cluster_prio["Cluster"] = "C" + contagem_cluster_prio["cluster"].astype(str)

    grafico_cluster_prio = px.bar(
        contagem_cluster_prio, x="Cluster", y="count", color="Prioridade",
        color_discrete_map=CORES_PRIORIDADE, barmode="stack",
        labels={"count": "Incidentes"},
    )
    grafico_cluster_prio.update_layout(**{**_layout_base, "height": 260})
    st.plotly_chart(grafico_cluster_prio, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 4 – RISCO DE OLA
# ══════════════════════════════════════════════════════════════════════════════
with aba4:
    st.markdown("""
    <div class='caixa-alerta'>
        <b>MONITOR DE OLA</b> — Acompanhe violações de acordos de nível operacional
        e identifique produtos com maior risco para intervenção preventiva.
    </div>
    """, unsafe_allow_html=True)

    # filtro só os incidentes que entraram no KPI
    df_kpi       = df[df["Entrou para KPI?"] == "SIM"].copy()
    total_kpi    = len(df_kpi)
    violacoes    = (df_kpi["KPI Violado?"] == "SIM").sum()
    pct_violacao = round(violacoes / total_kpi * 100, 2) if total_kpi > 0 else 0
    nao_violados = total_kpi - violacoes

    col_o1, col_o2, col_o3, col_o4 = st.columns(4)
    for coluna, valor, rotulo, cor in [
        (col_o1, total_kpi,          "ENTROU NO KPI",    CORES["ciano"]),
        (col_o2, violacoes,          "KPI VIOLADO",      CORES["vermelho"]),
        (col_o3, nao_violados,       "KPI CUMPRIDO",     CORES["sucesso"]),
        (col_o4, f"{pct_violacao}%", "TAXA DE VIOLAÇÃO", CORES["alerta"]),
    ]:
        with coluna:
            st.markdown(f"""<div class='cartao-metrica'>
                <div class='valor' style='color:{cor}'>{valor}</div>
                <div class='rotulo'>{rotulo}</div>
            </div>""", unsafe_allow_html=True)

    col_ola1, col_ola2 = st.columns(2)

    with col_ola1:
        st.markdown("<div class='titulo-secao'>▸ Tendência de Violação OLA (semanal)</div>", unsafe_allow_html=True)

        # agrupo por semana pra ter uma visão temporal da evolução da violação
        df_kpi["semana_dt"] = df_kpi["Aberto"].dt.to_period("W").apply(lambda r: r.start_time)
        ola_semanal = df_kpi.groupby("semana_dt").agg(
            entrou = ("entrou_kpi_bin", "count"),
            violou = ("kpi_violado_bin", "sum")
        ).reset_index()
        ola_semanal["pct"] = (ola_semanal["violou"] / ola_semanal["entrou"] * 100).round(2)

        # eixo duplo: barras = volume bruto, linha = percentual de violação
        grafico_ola = make_subplots(specs=[[{"secondary_y": True}]])
        grafico_ola.add_trace(go.Bar(
            x=ola_semanal["semana_dt"], y=ola_semanal["entrou"],
            name="Entrou KPI", marker_color=CORES["ciano"], opacity=0.35
        ), secondary_y=False)
        grafico_ola.add_trace(go.Scatter(
            x=ola_semanal["semana_dt"], y=ola_semanal["pct"],
            name="% Violado", line=dict(color=CORES["vermelho"], width=2),
            mode="lines+markers"
        ), secondary_y=True)
        grafico_ola.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color="#5E6680", size=10),
            height=300, margin=dict(t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1A2138", linecolor="#2A3050"),
        )
        grafico_ola.update_yaxes(gridcolor="#1A2138", secondary_y=False)
        grafico_ola.update_yaxes(title_text="% Violação OLA", secondary_y=True, gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(grafico_ola, use_container_width=True)

    with col_ola2:
        st.markdown("<div class='titulo-secao'>▸ Produtos com Maior Taxa de Violação OLA</div>", unsafe_allow_html=True)
        ola_produto = df_kpi.groupby("Produto").agg(
            entrou = ("entrou_kpi_bin", "count"),
            violou = ("kpi_violado_bin", "sum")
        ).reset_index()
        # mínimo 5 registros pra não distorcer percentual com amostras minúsculas
        ola_produto = ola_produto[ola_produto["entrou"] >= 5]
        ola_produto["taxa"] = (ola_produto["violou"] / ola_produto["entrou"] * 100).round(1)
        ola_produto = ola_produto.sort_values("taxa", ascending=False).head(12)

        grafico_ola_produto = px.bar(
            ola_produto, x="taxa", y="Produto", orientation="h",
            color="taxa", color_continuous_scale=["#00C1D4", "#FFA02F", "#E8003D"],
            labels={"taxa": "% Violação"},
        )
        grafico_ola_produto.update_layout(**{
            **_layout_base, "height": 300, "coloraxis_showscale": False
        })
        st.plotly_chart(grafico_ola_produto, use_container_width=True)

    # ── Tabela de risco ──────────────────────────────────────────────────────
    st.markdown("<div class='titulo-secao'>▸ Tabela de Risco Operacional — Produto × Prioridade</div>", unsafe_allow_html=True)
    matriz_risco = df_kpi.groupby(["Produto", "Prioridade"]).agg(
        total     = ("entrou_kpi_bin", "count"),
        violacoes = ("kpi_violado_bin", "sum"),
    ).reset_index()
    matriz_risco["taxa_violacao_%"] = (
        matriz_risco["violacoes"] / matriz_risco["total"] * 100
    ).round(1)

    # filtro mínimo de 3 registros e pego os 20 piores
    matriz_risco = (
        matriz_risco[matriz_risco["total"] >= 3]
        .sort_values("taxa_violacao_%", ascending=False)
        .head(20)
    )
    matriz_risco = matriz_risco.rename(columns={
        "total":           "Entrou KPI",
        "violacoes":       "Violações",
        "taxa_violacao_%": "Taxa Violação (%)"
    })

    def colorir_risco(val):
        # vermelho pra violação alta, âmbar pra moderada
        if isinstance(val, float):
            if val >= 10:
                return "color: #E8003D; font-weight: bold"
            elif val >= 5:
                return "color: #FFA02F"
        return ""

    st.dataframe(
        matriz_risco.style.map(colorir_risco, subset=["Taxa Violação (%)"]),
        use_container_width=True, hide_index=True,
    )

    # ── Recomendações ────────────────────────────────────────────────────────
    st.markdown("<div class='titulo-secao'>▸ Recomendações Operacionais</div>", unsafe_allow_html=True)
    produtos_alto_risco = matriz_risco[matriz_risco["Taxa Violação (%)"] >= 10]["Produto"].unique()
    hora_pico = df["hora"].value_counts().idxmax()
    dia_pico  = df["dia_semana"].value_counts().idxmax()

    recomendacoes = [
        f"🔴 **Atenção imediata** nos produtos: {', '.join(produtos_alto_risco[:5]) if len(produtos_alto_risco) > 0 else 'Nenhum crítico identificado'} — taxa de violação de OLA acima de 10%",
        f"⏰ **Pico de incidentes** ocorre às **{hora_pico}h** — reforce staffing nesse horário",
        f"📅 **Dia mais crítico da semana**: {dia_pico} — planeje escalas preventivas",
        f"🎯 **Prioridades P2 e P3** representam {round((df['Prioridade'].isin(['2 - Alta','3 - Média']).sum()/len(df))*100,1)}% dos incidentes — foco de monitoramento",
        "📊 **Execute clustering semanal** para detectar novos padrões de falha sistêmica antes que se tornem críticos",
    ]
    for recomendacao in recomendacoes:
        st.markdown(recomendacao)
