"""
PageGeneralOCR Pro - Askeri Tarih Araştırma Sistemi
Profesyonel dark tema arayüzü
"""
import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import datetime

from src.config import settings
# from src.agents.ingestion_agent import IngestionAgent  # OCR devre disi
from src.agents.rag_agent import RAGAgent
from src.services.vector_db_service import VectorDBService
from qdrant_client import QdrantClient

# =============================================================================
# SAYFA YAPILANDIRMASI
# =============================================================================
st.set_page_config(
    page_title="PageGeneral",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logo path
LOGO_PATH = "logo.png"

# Sabit font boyutu
FONT_SIZE = 16

# =============================================================================
# PROFESYONEL CSS - MODERN DARK TEMA
# =============================================================================
def get_custom_css():
    """GitHub/Discord tarzı modern dark tema CSS"""
    font_size = FONT_SIZE

    # Ana renkler - GitHub Dark teması
    bg_primary = "#0d1117"
    bg_secondary = "#161b22"
    bg_tertiary = "#21262d"
    bg_canvas = "#010409"

    # Border renkleri
    border_default = "#30363d"
    border_muted = "#21262d"

    # Text renkleri
    text_primary = "#e6edf3"
    text_secondary = "#8b949e"
    text_muted = "#6e7681"

    # Accent renkler
    accent_primary = "#238636"
    accent_secondary = "#1f6feb"
    accent_emphasis = "#58a6ff"
    accent_muted = "#388bfd26"

    # Uyarı renkleri
    success_fg = "#3fb950"
    warning_fg = "#d29922"
    danger_fg = "#f85149"

    return f"""
    <style>
        /* ===== GENEL RESET VE TEMEL YAPI ===== */
        .stApp {{
            background-color: {bg_primary};
        }}

        /* Ana container */
        .main .block-container {{
            padding: 2rem 3rem;
            max-width: 1400px;
        }}

        /* Temel font ayarlari */
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            font-size: {font_size}px;
            line-height: 1.5;
            color: {text_primary};
        }}

        /* ===== LOGO VE BASLIK ALANI ===== */
        .header-container {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
            padding: 1.5rem 0;
            margin-bottom: 1rem;
        }}

        .header-logo {{
            flex-shrink: 0;
        }}

        .header-text {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .header-title {{
            font-size: 2.5rem;
            font-weight: 600;
            color: {text_primary};
            margin: 0;
            letter-spacing: -0.5px;
        }}

        .header-subtitle {{
            font-size: 1.1rem;
            color: {text_secondary};
            margin: 0;
            font-weight: 400;
        }}

        .header-divider {{
            height: 1px;
            background: linear-gradient(90deg, {border_default} 0%, transparent 100%);
            margin: 0.5rem 0 2rem 0;
        }}

        /* ===== KARTLAR ===== */
        .info-card {{
            background-color: {bg_secondary};
            border: 1px solid {border_default};
            border-radius: 6px;
            padding: 1.25rem;
            margin: 1rem 0;
        }}

        .info-card h3 {{
            color: {text_primary};
            font-size: {font_size + 2}px;
            font-weight: 600;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {border_default};
        }}

        /* ===== ADIM GOSTERGESI ===== */
        .steps-container {{
            background-color: {bg_secondary};
            border: 1px solid {border_default};
            border-radius: 6px;
            padding: 1.25rem;
            margin: 1.5rem 0;
        }}

        .step {{
            display: flex;
            align-items: center;
            padding: 0.875rem 1rem;
            margin: 0.5rem 0;
            background-color: {bg_tertiary};
            border-radius: 6px;
            border-left: 3px solid {accent_primary};
            transition: background-color 0.15s ease;
        }}

        .step:hover {{
            background-color: {accent_muted};
        }}

        .step-number {{
            background-color: {accent_primary};
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: {font_size - 2}px;
            margin-right: 1rem;
            flex-shrink: 0;
        }}

        .step-text {{
            font-size: {font_size}px;
            color: {text_primary};
        }}

        .step-text strong {{
            color: {text_primary};
        }}

        /* ===== BUTONLAR ===== */
        .stButton > button {{
            font-size: {font_size}px !important;
            font-weight: 500 !important;
            padding: 0.625rem 1rem !important;
            border-radius: 6px !important;
            min-height: 44px !important;
            transition: all 0.15s ease !important;
            border: 1px solid {border_default} !important;
            background-color: {bg_tertiary} !important;
            color: {text_primary} !important;
        }}

        .stButton > button:hover {{
            background-color: {bg_secondary} !important;
            border-color: {text_muted} !important;
        }}

        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {{
            background-color: {accent_primary} !important;
            border-color: {accent_primary} !important;
            color: white !important;
        }}

        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover {{
            background-color: #2ea043 !important;
            border-color: #2ea043 !important;
        }}

        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {{
            background-color: {bg_secondary};
            border-right: 1px solid {border_default};
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding: 1.5rem 1rem;
        }}

        .sidebar-header {{
            font-size: {font_size + 2}px;
            font-weight: 600;
            color: {text_primary};
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {border_default};
        }}

        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stSlider label {{
            font-size: {font_size - 1}px !important;
            font-weight: 500 !important;
            color: {text_secondary} !important;
            margin-bottom: 0.25rem !important;
        }}

        /* ===== SELECT BOX ===== */
        .stSelectbox > div > div {{
            font-size: {font_size}px !important;
            background-color: {bg_tertiary} !important;
            border-color: {border_default} !important;
            border-radius: 6px !important;
            min-height: 40px !important;
        }}

        .stSelectbox > div > div:hover {{
            border-color: {text_muted} !important;
        }}

        /* ===== TEXT INPUT ===== */
        .stTextInput > div > div > input {{
            font-size: {font_size}px !important;
            background-color: {bg_tertiary} !important;
            border-color: {border_default} !important;
            border-radius: 6px !important;
            min-height: 40px !important;
            padding: 0.5rem 0.75rem !important;
            color: {text_primary} !important;
        }}

        .stTextInput > div > div > input:focus {{
            border-color: {accent_secondary} !important;
            box-shadow: 0 0 0 3px {accent_muted} !important;
        }}

        .stTextInput > div > div > input::placeholder {{
            color: {text_muted} !important;
        }}

        /* ===== CHAT INPUT ===== */
        .stChatInput > div {{
            border: 1px solid {border_default} !important;
            border-radius: 6px !important;
            background-color: {bg_secondary} !important;
        }}

        .stChatInput > div:focus-within {{
            border-color: {accent_secondary} !important;
            box-shadow: 0 0 0 3px {accent_muted} !important;
        }}

        .stChatInput textarea {{
            font-size: {font_size}px !important;
            color: {text_primary} !important;
            background-color: transparent !important;
        }}

        .stChatInput textarea::placeholder {{
            color: {text_muted} !important;
        }}

        /* ===== CHAT MESAJLARI ===== */
        [data-testid="stChatMessage"] {{
            background-color: {bg_secondary};
            border: 1px solid {border_default};
            border-radius: 6px;
            padding: 1rem;
            margin: 0.5rem 0;
            font-size: {font_size}px;
        }}

        /* Kullanici mesaji */
        [data-testid="stChatMessage"][data-testid*="user"] {{
            background-color: {bg_tertiary};
        }}

        /* ===== KAYNAKLAR KUTUSU ===== */
        .sources-box {{
            background-color: {bg_tertiary};
            border: 1px solid {border_default};
            border-radius: 6px;
            padding: 0.875rem;
            margin-top: 0.75rem;
            font-size: {font_size - 2}px;
        }}

        .sources-box strong {{
            color: {text_secondary};
        }}

        .source-tag {{
            display: inline-block;
            background-color: {accent_muted};
            color: {accent_emphasis};
            padding: 0.25rem 0.625rem;
            border-radius: 20px;
            margin: 0.25rem 0.25rem 0.25rem 0;
            font-size: {font_size - 3}px;
            font-weight: 500;
            border: 1px solid {accent_secondary}40;
        }}

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            background-color: transparent;
            border-bottom: 1px solid {border_default};
        }}

        .stTabs [data-baseweb="tab"] {{
            font-size: {font_size}px !important;
            font-weight: 500 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 0 !important;
            color: {text_secondary} !important;
            background-color: transparent !important;
            border-bottom: 2px solid transparent !important;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            color: {text_primary} !important;
            background-color: {bg_tertiary} !important;
        }}

        .stTabs [aria-selected="true"] {{
            color: {text_primary} !important;
            border-bottom-color: {accent_primary} !important;
        }}

        /* ===== DATAFRAME ===== */
        .stDataFrame {{
            border: 1px solid {border_default};
            border-radius: 6px;
            overflow: hidden;
        }}

        .stDataFrame > div {{
            font-size: {font_size - 1}px !important;
        }}

        /* ===== METRIKLER ===== */
        [data-testid="stMetricValue"] {{
            font-size: {font_size + 12}px !important;
            font-weight: 600 !important;
            color: {text_primary} !important;
        }}

        [data-testid="stMetricLabel"] {{
            font-size: {font_size - 2}px !important;
            color: {text_secondary} !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* ===== SLIDER ===== */
        .stSlider > div > div {{
            font-size: {font_size}px !important;
        }}

        .stSlider [data-baseweb="slider"] {{
            margin-top: 0.5rem;
        }}

        /* ===== UYARI KUTULARI ===== */
        .stAlert {{
            font-size: {font_size}px !important;
            padding: 0.875rem 1rem !important;
            border-radius: 6px !important;
            border-width: 1px !important;
        }}

        /* Info */
        [data-testid="stAlert"][data-baseweb="notification"] {{
            background-color: {accent_muted} !important;
            border-color: {accent_secondary}40 !important;
        }}

        /* ===== DIVIDER ===== */
        hr {{
            border: none;
            border-top: 1px solid {border_default};
            margin: 1.5rem 0;
        }}

        .stDivider {{
            background-color: {border_default};
        }}

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: {bg_primary};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {border_default};
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {text_muted};
        }}

        /* ===== FILE UPLOADER ===== */
        [data-testid="stFileUploader"] {{
            background-color: {bg_secondary};
            border: 1px dashed {border_default};
            border-radius: 6px;
            padding: 1rem;
        }}

        [data-testid="stFileUploader"]:hover {{
            border-color: {text_muted};
        }}

        /* ===== SPINNER ===== */
        .stSpinner > div {{
            border-top-color: {accent_primary} !important;
        }}

        /* ===== CAPTION ===== */
        .stCaption {{
            color: {text_muted} !important;
            font-size: {font_size - 2}px !important;
        }}

        /* ===== MARKDOWN BASLIKLAR ===== */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_primary} !important;
            font-weight: 600 !important;
        }}

        h3 {{
            font-size: {font_size + 4}px !important;
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
        }}

        /* ===== FOOTER ===== */
        .footer-container {{
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid {border_default};
        }}

        .footer-content {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 0.5rem;
            padding: 1.5rem 0;
        }}

        .footer-logo {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {text_primary};
            margin-bottom: 0.25rem;
        }}

        .footer-tagline {{
            font-size: {font_size - 1}px;
            color: {text_secondary};
            max-width: 400px;
        }}

        .footer-credit {{
            font-size: {font_size - 2}px;
            color: {text_muted};
            margin-top: 0.5rem;
        }}

        .footer-divider {{
            width: 60px;
            height: 2px;
            background: linear-gradient(90deg, transparent, {border_default}, transparent);
            margin: 0.75rem 0;
        }}

        /* ===== SECTION BASLIK ===== */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 1.5rem 0 1rem 0;
            font-size: {font_size + 2}px;
            font-weight: 600;
            color: {text_primary};
        }}

        .section-header-icon {{
            font-size: 1.25rem;
        }}

        /* ===== DOWNLOAD BUTTON ===== */
        .stDownloadButton > button {{
            font-size: {font_size - 1}px !important;
            background-color: {bg_tertiary} !important;
            border-color: {border_default} !important;
            color: {text_primary} !important;
        }}

        .stDownloadButton > button:hover {{
            background-color: {bg_secondary} !important;
            border-color: {text_muted} !important;
        }}

        /* ===== MOBIL UYUMLULUK ===== */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1rem !important;
                max-width: 100% !important;
            }}

            h1 {{
                font-size: 1.5rem !important;
            }}

            h3 {{
                font-size: 1.1rem !important;
            }}

            .steps-container {{
                padding: 0.75rem !important;
            }}

            .step {{
                padding: 0.75rem !important;
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }}

            .step-number {{
                width: 28px;
                height: 28px;
                font-size: 14px;
            }}

            .step-text {{
                font-size: 14px !important;
            }}

            .stButton > button {{
                font-size: 14px !important;
                padding: 0.5rem 1rem !important;
                min-height: 44px !important;
            }}

            [data-testid="stChatMessage"] {{
                padding: 0.75rem !important;
            }}

            .stChatInput textarea {{
                font-size: 16px !important;
            }}

            .footer-logo {{
                font-size: 1.1rem;
            }}

            .footer-tagline {{
                font-size: 13px;
            }}

            .footer-credit {{
                font-size: 12px;
            }}

            .source-tag {{
                font-size: 11px !important;
                padding: 3px 8px !important;
            }}

            [data-testid="stMetricValue"] {{
                font-size: 1.5rem !important;
            }}

            [data-testid="stMetricLabel"] {{
                font-size: 12px !important;
            }}
        }}

        @media (max-width: 480px) {{
            .main .block-container {{
                padding: 0.5rem !important;
            }}

            h1 {{
                font-size: 1.3rem !important;
            }}

            .steps-container {{
                padding: 0.5rem !important;
            }}

            .step {{
                padding: 0.5rem !important;
            }}
        }}
    </style>
    """

# =============================================================================
# SESSION STATE
# =============================================================================
if "inspector_messages" not in st.session_state:
    st.session_state.inspector_messages = []
if "inspector_data" not in st.session_state:
    st.session_state.inspector_data = []

# =============================================================================
# QDRANT CLIENT (Singleton - kilitleme sorununu onler)
# =============================================================================
@st.cache_resource
def init_qdrant():
    """Qdrant client'i bir kez olustur ve VectorDBService'e enjekte et"""
    import os

    # Lock dosyalarini temizle
    for lock_name in [".lock", ".qdrant_flock"]:
        lock_file = settings.QDRANT_PATH / lock_name
        if lock_file.exists():
            try:
                os.remove(lock_file)
            except:
                pass

    client = QdrantClient(path=str(settings.QDRANT_PATH))
    VectorDBService.set_client(client)
    return client

init_qdrant()

# =============================================================================
# AGENT'LAR
# =============================================================================
@st.cache_resource
def get_agents():
    return RAGAgent()

rag_agent = get_agents()

# =============================================================================
# SIDEBAR - AYARLAR
# =============================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-header">Kaynak Secimi</p>', unsafe_allow_html=True)

    books = ["Tum Kitaplar"] + rag_agent.get_ingested_books()
    selected_book = st.selectbox(
        "Kitap",
        books,
        help="Aramayi belirli bir kitapla sinirlayin"
    )

    units = ["Tum Birlikler"] + rag_agent.get_all_units()

    # Birlik arama
    unit_search = st.text_input(
        "Birlik Ara",
        placeholder="Orn: 57. Tumen",
        help="Birlik listesini filtrelemek icin yazin"
    )

    if unit_search:
        filtered_units = [u for u in units if unit_search.lower() in u.lower()]
        selected_unit = st.selectbox("Birlik Secin", filtered_units[:30])
    else:
        selected_unit = st.selectbox("Birlik Secin", units[:50])

    st.divider()

    # Istatistikler
    st.markdown('<p class="sidebar-header">Veritabani</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kitap", len(books) - 1)
    with col2:
        st.metric("Birlik", len(units) - 1)

# CSS uygula
st.markdown(get_custom_css(), unsafe_allow_html=True)

# =============================================================================
# ANA BASLIK
# =============================================================================
header_col1, header_col2 = st.columns([1, 11])
with header_col1:
    st.image(LOGO_PATH, width=90)
with header_col2:
    st.markdown("""
    <div style="padding: 0.75rem 0 0 0;">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 600; color: #e6edf3; letter-spacing: -0.5px;">PageGeneral</h1>
        <p style="margin: 0.35rem 0 0 0; font-size: 1rem; color: #8b949e;">Turk Kurtulus Savasi ve Askeri Tarih Belgeleri</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: none; border-top: 1px solid #30363d; margin: 1.5rem 0;'>", unsafe_allow_html=True)

# =============================================================================
# ANA SEKMELER
# =============================================================================
# OCR devre disi - sadece soru sor sekmesi aktif
tab1 = st.container()

# =============================================================================
# ANA ICERIK: SORU SOR
# =============================================================================
with tab1:
    # Kullanim adimlari
    st.markdown("""
    <div class="steps-container">
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-text"><strong>Kaynak Secin:</strong> Sol menuден kitap veya birlik secin</div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-text"><strong>Verileri Yukleyin:</strong> Asagidaki butona basin</div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-text"><strong>Soru Sorun:</strong> En alttaki kutuya sorunuzu yazin</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Veri yukleme bolumu
    st.markdown('<div class="section-header"><span class="section-header-icon">&#128229;</span> Veri Yukleme</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        limit = st.slider(
            "Kayit Sayisi",
            min_value=50,
            max_value=2000,
            value=200,
            step=50,
            help="Yuklenecek maksimum kayit sayisi"
        )

    with col2:
        st.write("")  # Bosluk
        st.write("")
        if st.button("VERILERI GETIR", type="primary", use_container_width=True):
            with st.spinner("Veriler yukleniyor..."):
                book_f = selected_book if selected_book != "Tum Kitaplar" else None
                unit_variations = rag_agent._get_unit_variations(selected_unit) if selected_unit != "Tum Birlikler" else []

                data = VectorDBService.browse_paragraphs(
                    book_filter=book_f,
                    unit_filter=unit_variations,
                    limit=limit
                )
                st.session_state.inspector_data = data
                st.session_state.inspector_messages = []
                st.success(f"{len(data)} kayit yuklendi!")

    with col3:
        st.write("")
        st.write("")
        if st.session_state.inspector_data:
            df = pd.DataFrame(st.session_state.inspector_data)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "CSV INDIR",
                csv,
                f"veriler_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )

    # Yuklenen veri tablosu - HER ZAMAN GORUNUR
    if st.session_state.inspector_data:
        st.markdown(f'<div class="section-header"><span class="section-header-icon">&#128203;</span> Yuklenen Veriler ({len(st.session_state.inspector_data)} kayit)</div>', unsafe_allow_html=True)
        df = pd.DataFrame(st.session_state.inspector_data)
        st.dataframe(df, height=350, use_container_width=True)

    st.divider()

    # =================================================================
    # SOHBET BOLUMU
    # =================================================================
    st.markdown('<div class="section-header"><span class="section-header-icon">&#128172;</span> Soru-Cevap</div>', unsafe_allow_html=True)

    if st.session_state.inspector_data:
        # Sohbet gecmisi
        for msg in st.session_state.inspector_messages:
            avatar = LOGO_PATH if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    sources_html = " ".join([f'<span class="source-tag">{s}</span>' for s in msg["sources"][:5]])
                    st.markdown(f'<div class="sources-box"><strong>Kaynaklar:</strong><br>{sources_html}</div>', unsafe_allow_html=True)

        # Soru girisi
        if prompt := st.chat_input("Sorunuzu buraya yazin..."):
            # Kullanici mesaji
            st.session_state.inspector_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Yanit
            with st.chat_message("assistant", avatar=LOGO_PATH):
                with st.spinner("Yanit hazirlaniyor..."):
                    book_f = selected_book if selected_book != "Tum Kitaplar" else None
                    unit_f = selected_unit if selected_unit != "Tum Birlikler" else None

                    # Gecmis
                    history = []
                    msgs = st.session_state.inspector_messages
                    for i in range(0, len(msgs) - 1, 2):
                        if i + 1 < len(msgs):
                            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                                history.append((msgs[i]["content"], msgs[i + 1]["content"]))

                    session_id = f"st_{unit_f or 'default'}"

                    answer, sources, timing = rag_agent.chat_with_context(
                        prompt,
                        history=history[-4:],
                        book_filter=book_f,
                        unit_filter=unit_f,
                        context_data=st.session_state.inspector_data,
                        session_id=session_id
                    )

                    st.markdown(answer)

                    if sources:
                        sources_html = " ".join([f'<span class="source-tag">{s}</span>' for s in sources[:5]])
                        st.markdown(f'<div class="sources-box"><strong>Kaynaklar:</strong><br>{sources_html}</div>', unsafe_allow_html=True)

                    st.caption(f"Yanit suresi: {timing.get('total', 0):.1f} saniye")

            st.session_state.inspector_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
    else:
        st.info("Soru sormak icin once yukaridaki 'VERILERI GETIR' butonuna tiklayin.")

# =============================================================================
# SEKME 2: PDF YUKLE (DEVRE DISI)
# =============================================================================
# OCR modulu su an devre disi birakilmistir.

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer-container">
    <div class="footer-content">
        <div class="footer-logo">PageGeneral</div>
        <div class="footer-divider"></div>
        <div class="footer-tagline">Turk Kurtulus Savasi ve Askeri Tarih Belgeleri</div>
        <div class="footer-credit">Baran Can Ercan tarafindan gelistirilmistir.</div>
    </div>
</div>
""", unsafe_allow_html=True)
