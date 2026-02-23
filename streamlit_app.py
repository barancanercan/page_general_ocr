"""
Profesyonel Streamlit Arayüzü - PageGeneralOCR Pro
Gradio versiyonuyla aynı mantık: Veri Müfettişi + PDF Yükle
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

# Sayfa yapılandırması
st.set_page_config(
    page_title="PageGeneralOCR Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .source-badge {
        background-color: #e9ecef;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state başlat
if "inspector_messages" not in st.session_state:
    st.session_state.inspector_messages = []
if "inspector_data" not in st.session_state:
    st.session_state.inspector_data = []

# Agent'ları başlat
@st.cache_resource
def get_agents():
    return IngestionAgent(), RAGAgent()

ingestion_agent, rag_agent = get_agents()

# Başlık
st.markdown('<h1 class="main-header">📚 PageGeneralOCR Pro</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")

    # Kitap ve birlik seçimi
    books = ["Tüm Kitaplar"] + rag_agent.get_ingested_books()
    selected_book = st.selectbox("📖 Kitap", books)

    units = ["Tüm Birlikler"] + rag_agent.get_all_units()
    unit_search = st.text_input("🔍 Birlik Ara", placeholder="Örn: 57. Tümen")

    if unit_search:
        filtered_units = [u for u in units if unit_search.lower() in u.lower()]
        selected_unit = st.selectbox("🎯 Birlik", filtered_units[:50])
    else:
        selected_unit = st.selectbox("🎯 Birlik", units[:50])

    st.divider()

    # İstatistikler
    st.header("📊 İstatistikler")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kitaplar", len(books) - 1)
    with col2:
        st.metric("Birlikler", len(units) - 1)

# Ana sekmeler (Gradio ile aynı: Veri Müfettişi + PDF Yükle)
tab1, tab2 = st.tabs(["🧐 Veri Müfettişi", "📤 PDF Yükle"])

# --- VERİ MÜFETTİŞİ SEKMESİ ---
with tab1:
    st.markdown("""
    <div class="info-box">
    <strong>Adımlar:</strong> 1) Birlik/Kitap seçin → 2) 'Verileri Getir' butonuna basın → 3) Aşağıdaki sohbet kutusundan birlik hakkında sorular sorun.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("📋 Veri Kontrolü")
        limit = st.slider("Satır sayısı", 10, 500, 50, 10)

        if st.button("📥 Verileri Getir", type="primary", use_container_width=True):
            with st.spinner("Veriler yükleniyor..."):
                book_f = selected_book if selected_book != "Tüm Kitaplar" else None
                unit_variations = rag_agent._get_unit_variations(selected_unit) if selected_unit != "Tüm Birlikler" else []

                data = VectorDBService.browse_paragraphs(
                    book_filter=book_f,
                    unit_filter=unit_variations,
                    limit=limit
                )
                st.session_state.inspector_data = data
                st.session_state.inspector_messages = []  # Yeni veri = yeni sohbet
                st.success(f"{len(data)} kayıt yüklendi")

    with col2:
        if st.session_state.inspector_data:
            df = pd.DataFrame(st.session_state.inspector_data)
            st.dataframe(df, use_container_width=True, height=300)

            # CSV indir
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 CSV İndir",
                csv,
                f"veriler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.info("Verileri görmek için soldaki 'Verileri Getir' butonuna tıklayın.")

    # Sohbet bölümü
    st.divider()
    st.subheader("💬 Birlik Verileriyle Sohbet")

    if st.session_state.inspector_data:
        # Sohbet geçmişi
        for msg in st.session_state.inspector_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption("**Kaynaklar:** " + " | ".join(msg["sources"]))

        # Sohbet girişi
        if prompt := st.chat_input("Bu birlik hakkında soru sorun..."):
            # Kullanıcı mesajı
            st.session_state.inspector_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Yanıt
            with st.chat_message("assistant"):
                with st.spinner("Analiz ediliyor..."):
                    book_f = selected_book if selected_book != "Tüm Kitaplar" else None
                    unit_f = selected_unit if selected_unit != "Tüm Birlikler" else None

                    # Geçmiş mesajları parse et
                    history = []
                    msgs = st.session_state.inspector_messages
                    for i in range(0, len(msgs) - 1, 2):
                        if i + 1 < len(msgs):
                            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                                history.append((msgs[i]["content"], msgs[i + 1]["content"]))

                    # Session ID oluştur
                    session_id = f"insp_{unit_f or 'default'}"

                    # chat_with_context kullan (Gradio ile aynı)
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
                        st.caption("**Kaynaklar:** " + " | ".join(sources))

                    st.caption(f"⏱️ LLM: {timing.get('llm', 0):.2f}s | Toplam: {timing.get('total', 0):.2f}s")

            st.session_state.inspector_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
    else:
        st.warning("Sohbet etmek için önce 'Verileri Getir' butonuna tıklayarak birlik verilerini yükleyin.")

# --- PDF YÜKLE SEKMESİ ---
with tab2:
    st.subheader("📤 PDF Yükle ve İşle")

    uploaded_file = st.file_uploader("PDF dosyası seçin", type=["pdf"])

    if uploaded_file:
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("🚀 İşle ve Veritabanına Ekle", type="primary"):
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
                log_area.text_area("İşlem Günlüğü", "\n".join(logs[-20:]), height=200)

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
                st.error(f"Hata: {e}")
            finally:
                os.unlink(tmp_path)
                st.cache_resource.clear()

# Footer
st.divider()
st.caption("PageGeneralOCR Pro - Askeri Tarih Araştırma Sistemi")
