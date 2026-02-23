import logging
import gradio as gr
import pandas as pd
import tempfile
from functools import lru_cache
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from src.config import settings
from src.agents.ingestion_agent import IngestionAgent
from src.agents.rag_agent import RAGAgent
from src.services.vector_db_service import VectorDBService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_last_cache_time = None
_cached_choices = None
CACHE_DURATION = timedelta(minutes=5)

def _parse_history(history: List[dict]) -> List[Tuple[str, str]]:
    pair_history = []
    i = 0
    while i < len(history) - 1:
        if history[i].get("role") == "user" and history[i + 1].get("role") == "assistant":
            user_msg = _extract_text(history[i]["content"])
            assistant_msg = _extract_text(history[i + 1]["content"])
            pair_history.append((user_msg, assistant_msg))
        i += 1
    return pair_history

def _normalize_filter(value, all_option):
    if value and value != all_option:
        return value
    return None

ingestion_agent = IngestionAgent()
rag_agent = RAGAgent()

def handle_upload(pdf_file) -> Tuple[str, dict, dict, list]:
    if pdf_file is None:
        return "Lütfen bir PDF dosyası seçin.", gr.update(), gr.update(), []

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    if not pdf_path.lower().endswith('.pdf'):
        return "Sadece PDF dosyaları kabul edilir.", gr.update(), gr.update(), []
    
    messages = []
    def on_progress(msg: str):
        messages.append(msg)

    result = ingestion_agent.ingest_pdf(pdf_path, max_pages=0, progress_callback=on_progress)

    status_icon = {"ok": "✓", "skipped": "⚠", "error": "✗"}.get(result["status"], "?")
    summary = f"{status_icon} {result['message']}"

    final_status = "\n".join(messages) + f"\n\n{summary}"
    
    book_update, unit_update, all_units = refresh_choices()
    
    return final_status, book_update, unit_update, all_units

def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)

def handle_chat(message: str, history: List[dict], book_filter: Optional[str] = None, unit_filter: Optional[str] = None) -> Tuple[str, str, str]:
    if not message.strip():
        return "", "", ""
    message = message.strip()[:2000]

    pair_history = _parse_history(history)

    bfilter = _normalize_filter(book_filter, "Tüm Kitaplar")
    ufilter = _normalize_filter(unit_filter, "Tüm Birlikler")

    answer, sources, timing = rag_agent.chat(message, history=pair_history, book_filter=bfilter, unit_filter=ufilter)

    sources_text = ""
    if sources:
        sources_text = "**Kullanılan Kaynaklar:**\n" + "\n".join(f"- {s}" for s in sources)

    timing_text = (
        f"⏱️ Arama: {timing.get('search', 0):.2f}s | "
        f"Yeniden Sıralama: {timing.get('rerank', 0):.2f}s | "
        f"LLM: {timing.get('llm', 0):.2f}s | "
        f"Toplam: {timing.get('total', 0):.2f}s"
    )

    return answer, sources_text, timing_text

def refresh_choices(force_refresh: bool = False) -> Tuple[dict, dict, list]:
    global _last_cache_time, _cached_choices
    
    if not force_refresh and _cached_choices and _last_cache_time:
        if datetime.now() - _last_cache_time < CACHE_DURATION:
            return _cached_choices
    
    books = rag_agent.get_ingested_books()
    book_choices = ["Tüm Kitaplar"] + books
    
    # Birimleri normalize et ve yinelenenleri kaldır
    raw_units = rag_agent.get_all_units()
    seen = set()
    unique_units = []
    for unit in raw_units:
        normalized = unit.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_units.append(unit)
    units = sorted(unique_units)
    unit_choices_initial = ["Tüm Birlikler"] + units
    
    logger.info(f"Refreshed choices. Books: {len(books)}, Units: {len(units)}")
    
    _cached_choices = (
        gr.update(choices=book_choices, value="Tüm Kitaplar"), 
        gr.update(choices=unit_choices_initial, value="Tüm Birlikler"),
        units
    )
    _last_cache_time = datetime.now()
    return _cached_choices

def filter_units(search_text: str, all_units: list) -> dict:
    if not all_units:
        return gr.update(choices=["Tüm Birlikler"])
        
    if not search_text:
        # Başlangıçta da yinelenenleri temizle
        seen = set()
        unique = []
        for u in all_units[:50]:
            if u.lower() not in seen:
                seen.add(u.lower())
                unique.append(u)
        return gr.update(choices=["Tüm Birlikler"] + unique)
    
    search_lower = search_text.lower()
    seen = set()
    filtered = []
    for u in all_units:
        if search_lower in u.lower():
            if u.lower() not in seen:
                seen.add(u.lower())
                filtered.append(u)
    limited_results = filtered[:100]
    
    return gr.update(choices=["Tüm Birlikler"] + limited_results, value="Tüm Birlikler" if not limited_results else limited_results[0])

def inspect_data(book_filter, unit_filter, limit):
    unit_variations = rag_agent._get_unit_variations(unit_filter)

    data = VectorDBService.browse_paragraphs(
        book_filter=book_filter,
        unit_filter=unit_variations,
        limit=int(limit)
    )
    
    if not data:
        return pd.DataFrame(columns=["Kitap", "Sayfa", "Birlikler", "Metin"]), None, []
        
    df = pd.DataFrame(data)
    
    csv_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig") as tmp:
            df.to_csv(tmp.name, index=False)
            csv_path = tmp.name
    except Exception as e:
        logger.error(f"CSV oluşturma hatası: {e}")
        return df, None, data
    return df, csv_path, data


def handle_inspector_chat(message: str, history: List[dict], book_filter: Optional[str] = None, unit_filter: Optional[str] = None, stored_data: Optional[List[dict]] = None) -> Tuple[str, str, str]:
    if not message.strip():
        return "", "", ""
    
    if not stored_data:
        return "Önce 'Verileri Getir' butonuna tıklayarak birlik verilerini yükleyin.", "", ""

    pair_history = _parse_history(history)

    unit_name = _normalize_filter(unit_filter, "Tüm Birlikler")
    session_id = f"insp_{unit_name or 'default'}"
    
    answer, sources, timing = rag_agent.chat_with_context(
        message, 
        history=pair_history, 
        book_filter=book_filter, 
        unit_filter=unit_name, 
        context_data=stored_data,
        session_id=session_id
    )

    sources_text = ""
    if sources:
        sources_text = "**Kullanılan Kaynaklar:**\n" + "\n".join(f"- {s}" for s in sources)

    timing_text = (
        f"⏱️ LLM: {timing.get('llm', 0):.2f}s | "
        f"Toplam: {timing.get('total', 0):.2f}s"
    )

    return answer, sources_text, timing_text

def build_ui() -> gr.Blocks:
    with gr.Blocks(title=settings.GRADIO_TITLE) as app:
        gr.Markdown(f"# {settings.GRADIO_TITLE}")
        
        all_units_state = gr.State([])
        inspector_data_state = gr.State([])
        
        with gr.Tabs():
            # --- SEKME 1: VERİ MÜFETTİŞİ ---
            with gr.Tab("🧐 Veri Müfettişi"):
                gr.Markdown("**Adımlar:** 1) Birlik/Kitap seçin → 2) 'Verileri Getir' butonuna basın → 3) Aşağıdaki sohbet kutusundan birlik hakkında sorular sorun.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Filtreleme")
                        insp_unit_search = gr.Textbox(label="🔍 Birlik Ara", placeholder="Örn: 57. Tümen")
                        insp_unit_filter = gr.Dropdown(label="🎯 Birlik Seç", choices=["Tüm Birlikler"], value="Tüm Birlikler", allow_custom_value=True)
                        insp_book_filter = gr.Dropdown(label="📚 Kitap Seç", choices=["Tüm Kitaplar"], value="Tüm Kitaplar")
                        insp_limit = gr.Slider(
                            label="Satır Sayısı", 
                            minimum=10, 
                            maximum=500, 
                            value=50, 
                            step=10, 
                            info="Daha fazla veri = daha uzun yükleme süresi"
                        )
                        insp_btn = gr.Button("Verileri Getir", variant="primary")
                        gr.Markdown("*Verileri getirdikten sonra sohbet edebilirsiniz.")
                    
                    with gr.Column(scale=3):
                        data_table = gr.Dataframe(
                            headers=["Kitap", "Sayfa", "Birlikler", "Metin"],
                            datatype=["str", "str", "str", "str"],
                            wrap=True,
                            interactive=False
                        )
                        download_btn = gr.File(label="📥 İndir (CSV)")
                
                gr.Markdown("---")
                gr.Markdown("### 💬 Birlik Verileriyle Sohbet")
                
                with gr.Row():
                    inspector_chatbot = gr.Chatbot(label="Sohbet", height=350)
                
                with gr.Row():
                    inspector_chat_input = gr.Textbox(
                        label="Sorunuzu yazın",
                        placeholder="Örn: Bu tümen hangi cephelerde savaştı?",
                        scale=4,
                    )
                    inspector_chat_btn = gr.Button("Gönder", variant="primary", scale=1)
                
                with gr.Accordion("Detaylar", open=False):
                    inspector_timing_box = gr.Textbox(label="Performans", interactive=False, lines=1)
                    inspector_sources_box = gr.Markdown(label="Kaynaklar")

            # --- SEKME 2: YÜKLEME ---
            with gr.Tab("📂 Belge Yükle"):
                pdf_input = gr.File(label="PDF Yükle", file_types=[".pdf"])
                upload_btn = gr.Button("İşle ve Veritabanına Ekle", variant="primary")
                upload_status = gr.Textbox(label="Durum", lines=10)

        # --- EVENT HANDLERS ---

        def _inspector_respond(message, history, book_filter, unit_filter, stored_data):
            if not message.strip(): return history, "", "", ""
            current_history = history if history else []
            answer, sources_text, timing_text = handle_inspector_chat(message, current_history, book_filter, unit_filter, stored_data)
            current_history.append({"role": "user", "content": message})
            current_history.append({"role": "assistant", "content": answer})
            return current_history, "", timing_text, sources_text

        def _update_inspector_data(book_filter, unit_filter, limit):
            df, csv_path, data = inspect_data(book_filter, unit_filter, limit)
            return df, csv_path, data

        # Inspector Events
        insp_unit_search.change(fn=filter_units, inputs=[insp_unit_search, all_units_state], outputs=[insp_unit_filter])
        
        # Get Data
        insp_btn.click(
            fn=_update_inspector_data, 
            inputs=[insp_book_filter, insp_unit_filter, insp_limit], 
            outputs=[data_table, download_btn, inspector_data_state]
        )
        
        # Chat
        inspector_chat_btn.click(
            fn=_inspector_respond, 
            inputs=[inspector_chat_input, inspector_chatbot, insp_book_filter, insp_unit_filter, inspector_data_state], 
            outputs=[inspector_chatbot, inspector_chat_input, inspector_timing_box, inspector_sources_box]
        )
        inspector_chat_input.submit(
            fn=_inspector_respond, 
            inputs=[inspector_chat_input, inspector_chatbot, insp_book_filter, insp_unit_filter, inspector_data_state], 
            outputs=[inspector_chatbot, inspector_chat_input, inspector_timing_box, inspector_sources_box]
        )

        # Upload
        upload_btn.click(fn=handle_upload, inputs=[pdf_input], outputs=[upload_status, insp_book_filter, insp_unit_filter, all_units_state])
        
        # Init - Force refresh to apply new deduplication logic
        app.load(fn=lambda: refresh_choices(force_refresh=True), outputs=[insp_book_filter, insp_unit_filter, all_units_state])

    return app

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_port=settings.GRADIO_PORT, share=False)
