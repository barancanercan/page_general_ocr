"""
PageGeneralOCR Pro - Askeri Tarih Araştırma Sistemi
Yaşlı kullanıcılar için optimize edilmiş profesyonel arayüz
"""
import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import datetime

from src.config import settings
from src.agents.ingestion_agent import IngestionAgent
from src.agents.rag_agent import RAGAgent
from src.services.vector_db_service import VectorDBService
from qdrant_client import QdrantClient

# =============================================================================
# SAYFA YAPILANDIRMASI
# =============================================================================
st.set_page_config(
    page_title="Askeri Tarih Araştırma",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROFESYONEL CSS - YAŞLI KULLANICILAR İÇİN OPTİMİZE
# =============================================================================
def get_custom_css(font_size: int = 18, high_contrast: bool = False):
    """Dinamik CSS oluştur - font boyutu ve kontrast moduna göre"""

    # Renk şeması
    if high_contrast:
        bg_color = "#000000"
        text_color = "#FFFFFF"
        card_bg = "#1a1a1a"
        border_color = "#FFFFFF"
        primary_color = "#4da6ff"
        secondary_bg = "#2d2d2d"
    else:
        bg_color = "#f5f7fa"
        text_color = "#1a1a2e"
        card_bg = "#ffffff"
        border_color = "#d1d5db"
        primary_color = "#1e3a5f"
        secondary_bg = "#e8ecf1"

    return f"""
    <style>
        /* ===== GENEL YAPI ===== */
        .stApp {{
            background-color: {bg_color};
        }}

        /* Tüm metinler için temel font */
        html, body, [class*="css"] {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: {font_size}px;
            line-height: 1.6;
            color: {text_color};
        }}

        /* ===== BAŞLIK ===== */
        .main-header {{
            background: linear-gradient(135deg, {primary_color} 0%, #2d5a87 100%);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .main-header h1 {{
            font-size: {font_size + 14}px;
            font-weight: 700;
            margin: 0;
            letter-spacing: 0.5px;
        }}

        .main-header p {{
            font-size: {font_size + 2}px;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }}

        /* ===== KARTLAR ===== */
        .info-card {{
            background-color: {card_bg};
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        .info-card h3 {{
            color: {primary_color};
            font-size: {font_size + 4}px;
            margin-bottom: 1rem;
            border-bottom: 2px solid {primary_color};
            padding-bottom: 0.5rem;
        }}

        /* ===== ADIM GÖSTERGESİ ===== */
        .steps-container {{
            background-color: {secondary_bg};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }}

        .step {{
            display: flex;
            align-items: center;
            padding: 1rem;
            margin: 0.5rem 0;
            background-color: {card_bg};
            border-radius: 8px;
            border-left: 4px solid {primary_color};
        }}

        .step-number {{
            background-color: {primary_color};
            color: white;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: {font_size + 2}px;
            margin-right: 1rem;
            flex-shrink: 0;
        }}

        .step-text {{
            font-size: {font_size}px;
            color: {text_color};
        }}

        /* ===== BUTONLAR ===== */
        .stButton > button {{
            font-size: {font_size + 2}px !important;
            padding: 0.8rem 2rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            min-height: 56px !important;
            transition: all 0.2s ease !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .stButton > button[kind="primary"] {{
            background-color: {primary_color} !important;
            border: none !important;
        }}

        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {{
            background-color: {card_bg};
            padding: 1rem;
        }}

        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stSlider label {{
            font-size: {font_size}px !important;
            font-weight: 600 !important;
            color: {text_color} !important;
        }}

        /* ===== SELECT BOX ===== */
        .stSelectbox > div > div {{
            font-size: {font_size}px !important;
            min-height: 48px !important;
        }}

        /* ===== TEXT INPUT ===== */
        .stTextInput > div > div > input {{
            font-size: {font_size}px !important;
            min-height: 48px !important;
            padding: 0.75rem !important;
        }}

        /* ===== CHAT INPUT ===== */
        .stChatInput > div {{
            border: 2px solid {primary_color} !important;
            border-radius: 12px !important;
        }}

        .stChatInput textarea {{
            font-size: {font_size + 2}px !important;
            min-height: 60px !important;
        }}

        /* ===== CHAT MESAJLARI ===== */
        [data-testid="stChatMessage"] {{
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1rem;
            margin: 0.75rem 0;
            font-size: {font_size}px;
        }}

        /* ===== KAYNAKLAR ===== */
        .sources-box {{
            background-color: {secondary_bg};
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            font-size: {font_size - 2}px;
        }}

        .source-tag {{
            display: inline-block;
            background-color: {primary_color};
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            margin: 0.25rem;
            font-size: {font_size - 2}px;
        }}

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            font-size: {font_size + 2}px !important;
            font-weight: 600 !important;
            padding: 1rem 2rem !important;
            border-radius: 8px 8px 0 0 !important;
        }}

        /* ===== DATAFRAME ===== */
        .stDataFrame {{
            font-size: {font_size - 1}px !important;
        }}

        /* ===== METRİKLER ===== */
        [data-testid="stMetricValue"] {{
            font-size: {font_size + 8}px !important;
            font-weight: 700 !important;
            color: {primary_color} !important;
        }}

        [data-testid="stMetricLabel"] {{
            font-size: {font_size}px !important;
        }}

        /* ===== SLIDER ===== */
        .stSlider > div > div {{
            font-size: {font_size}px !important;
        }}

        /* ===== UYARI KUTULARI ===== */
        .stAlert {{
            font-size: {font_size}px !important;
            padding: 1rem !important;
            border-radius: 8px !important;
        }}

        /* ===== FOOTER ===== */
        .footer {{
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            border-top: 2px solid {border_color};
            color: {text_color};
            opacity: 0.7;
            font-size: {font_size - 2}px;
        }}

        /* ===== DIVIDER ===== */
        hr {{
            border: none;
            border-top: 2px solid {border_color};
            margin: 2rem 0;
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
if "font_size" not in st.session_state:
    st.session_state.font_size = 18
if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False

# =============================================================================
# QDRANT CLIENT (Singleton - kilitleme sorununu önler)
# =============================================================================
@st.cache_resource
def init_qdrant():
    """Qdrant client'ı bir kez oluştur ve VectorDBService'e enjekte et"""
    import os
    lock_file = settings.QDRANT_PATH / ".lock"

    # Lock dosyası varsa sil
    if lock_file.exists():
        try:
            os.remove(lock_file)
        except:
            pass

    client = QdrantClient(path=str(settings.QDRANT_PATH))
    VectorDBService._client = client
    return client

init_qdrant()

# =============================================================================
# AGENT'LAR
# =============================================================================
@st.cache_resource
def get_agents():
    return IngestionAgent(), RAGAgent()

ingestion_agent, rag_agent = get_agents()

# =============================================================================
# SIDEBAR - AYARLAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Görünüm Ayarları")

    # Font boyutu
    font_size = st.slider(
        "📏 Yazı Boyutu",
        min_value=14,
        max_value=28,
        value=st.session_state.font_size,
        step=2,
        help="Tüm metinlerin boyutunu ayarlayın"
    )
    st.session_state.font_size = font_size

    # Yüksek kontrast
    high_contrast = st.checkbox(
        "🌙 Yüksek Kontrast Modu",
        value=st.session_state.high_contrast,
        help="Görme zorluğu için koyu tema"
    )
    st.session_state.high_contrast = high_contrast

    st.divider()

    # Kitap ve birlik seçimi
    st.markdown("## 📖 Kaynak Seçimi")

    books = ["Tüm Kitaplar"] + rag_agent.get_ingested_books()
    selected_book = st.selectbox(
        "Kitap",
        books,
        help="Aramayı belirli bir kitapla sınırlayın"
    )

    units = ["Tüm Birlikler"] + rag_agent.get_all_units()

    # Birlik arama
    unit_search = st.text_input(
        "🔍 Birlik Ara",
        placeholder="Örn: 57. Tümen",
        help="Birlik listesini filtrelemek için yazın"
    )

    if unit_search:
        filtered_units = [u for u in units if unit_search.lower() in u.lower()]
        selected_unit = st.selectbox("Birlik Seçin", filtered_units[:30])
    else:
        selected_unit = st.selectbox("Birlik Seçin", units[:50])

    st.divider()

    # İstatistikler
    st.markdown("## 📊 Veritabanı")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kitap", len(books) - 1)
    with col2:
        st.metric("Birlik", len(units) - 1)

# CSS uygula
st.markdown(get_custom_css(font_size, high_contrast), unsafe_allow_html=True)

# =============================================================================
# ANA BAŞLIK
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>📚 Askeri Tarih Araştırma Sistemi</h1>
    <p>Türk Kurtuluş Savaşı ve Askeri Tarih Belgeleri</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# ANA SEKMELER
# =============================================================================
tab1, tab2 = st.tabs(["🔍 Soru Sor", "📤 PDF Yükle"])

# =============================================================================
# SEKME 1: SORU SOR
# =============================================================================
with tab1:
    # Kullanım adımları
    st.markdown("""
    <div class="steps-container">
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-text"><strong>Kaynak Seçin:</strong> Sol menüden kitap veya birlik seçin</div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-text"><strong>Verileri Yükleyin:</strong> Aşağıdaki butona basın</div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-text"><strong>Soru Sorun:</strong> En alttaki kutuya sorunuzu yazın</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Veri yükleme bölümü
    st.markdown("### 📥 Veri Yükleme")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        limit = st.slider(
            "Kayıt Sayısı",
            min_value=50,
            max_value=2000,
            value=200,
            step=50,
            help="Yüklenecek maksimum kayıt sayısı"
        )

    with col2:
        st.write("")  # Boşluk
        st.write("")
        if st.button("📥 VERİLERİ GETİR", type="primary", use_container_width=True):
            with st.spinner("Veriler yükleniyor..."):
                book_f = selected_book if selected_book != "Tüm Kitaplar" else None
                unit_variations = rag_agent._get_unit_variations(selected_unit) if selected_unit != "Tüm Birlikler" else []

                data = VectorDBService.browse_paragraphs(
                    book_filter=book_f,
                    unit_filter=unit_variations,
                    limit=limit
                )
                st.session_state.inspector_data = data
                st.session_state.inspector_messages = []
                st.success(f"✅ {len(data)} kayıt yüklendi!")

    with col3:
        st.write("")
        st.write("")
        if st.session_state.inspector_data:
            df = pd.DataFrame(st.session_state.inspector_data)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "💾 CSV İNDİR",
                csv,
                f"veriler_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )

    # Yüklenen veri tablosu - HER ZAMAN GÖRÜNÜR
    if st.session_state.inspector_data:
        st.markdown(f"### 📋 Yüklenen Veriler ({len(st.session_state.inspector_data)} kayıt)")
        df = pd.DataFrame(st.session_state.inspector_data)
        st.dataframe(df, height=350, use_container_width=True)

    st.divider()

    # =================================================================
    # SOHBET BÖLÜMÜ
    # =================================================================
    st.markdown("### 💬 Soru-Cevap")

    if st.session_state.inspector_data:
        # Sohbet geçmişi
        for msg in st.session_state.inspector_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    sources_html = " ".join([f'<span class="source-tag">{s}</span>' for s in msg["sources"][:5]])
                    st.markdown(f'<div class="sources-box"><strong>📚 Kaynaklar:</strong><br>{sources_html}</div>', unsafe_allow_html=True)

        # Soru girişi
        if prompt := st.chat_input("Sorunuzu buraya yazın..."):
            # Kullanıcı mesajı
            st.session_state.inspector_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Yanıt
            with st.chat_message("assistant"):
                with st.spinner("Yanıt hazırlanıyor..."):
                    book_f = selected_book if selected_book != "Tüm Kitaplar" else None
                    unit_f = selected_unit if selected_unit != "Tüm Birlikler" else None

                    # Geçmiş
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
                        st.markdown(f'<div class="sources-box"><strong>📚 Kaynaklar:</strong><br>{sources_html}</div>', unsafe_allow_html=True)

                    st.caption(f"⏱️ Yanıt süresi: {timing.get('total', 0):.1f} saniye")

            st.session_state.inspector_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
    else:
        st.info("👆 Soru sormak için önce yukarıdaki **'VERİLERİ GETİR'** butonuna tıklayın.")

# =============================================================================
# SEKME 2: PDF YÜKLE
# =============================================================================
with tab2:
    st.markdown("""
    <div class="info-card">
        <h3>📤 Yeni PDF Ekle</h3>
        <p>Veritabanına yeni bir askeri tarih belgesi eklemek için PDF dosyası yükleyin.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "PDF dosyası seçin",
        type=["pdf"],
        help="Maksimum 200MB boyutunda PDF dosyası"
    )

    if uploaded_file:
        st.success(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.1f} MB)")

        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("🚀 İŞLE VE EKLE", type="primary", use_container_width=True):
                # Geçici dosyaya kaydet
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                progress_bar = st.progress(0)
                status_text = st.empty()
                log_area = st.empty()

                logs = []

                def on_progress(msg: str):
                    logs.append(msg)
                    log_area.text_area("İşlem Günlüğü", "\n".join(logs[-15:]), height=250)

                try:
                    result = ingestion_agent.ingest_pdf(tmp_path, max_pages=0, progress_callback=on_progress)

                    if result["status"] == "ok":
                        st.success(f"✅ {result['message']}")
                        progress_bar.progress(100)
                    elif result["status"] == "skipped":
                        st.warning(f"⚠️ {result['message']}")
                    else:
                        st.error(f"❌ {result['message']}")

                except Exception as e:
                    st.error(f"Hata oluştu: {e}")
                finally:
                    os.unlink(tmp_path)
                    st.cache_resource.clear()

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    <strong>Askeri Tarih Araştırma Sistemi</strong> | PageGeneralOCR Pro<br>
    Türk Kurtuluş Savaşı ve Askeri Tarih Belgeleri Veritabanı
</div>
""", unsafe_allow_html=True)
