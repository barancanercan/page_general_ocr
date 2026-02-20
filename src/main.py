import logging
import gradio as gr
import pandas as pd
import tempfile
from typing import List, Tuple, Optional

from src.config import settings
from src.agents.ingestion_agent import IngestionAgent
from src.agents.rag_agent import RAGAgent
from src.services.vector_db_service import VectorDBService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Agents
ingestion_agent = IngestionAgent()
rag_agent = RAGAgent()

def handle_upload(pdf_file) -> Tuple[str, dict, dict, list]:
    if pdf_file is None:
        return "Lutfen bir PDF dosyasi secin.", gr.update(), gr.update(), []

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    
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

    pair_history = []
    i = 0
    while i < len(history) - 1:
        if history[i].get("role") == "user" and history[i + 1].get("role") == "assistant":
            pair_history.append((
                _extract_text(history[i]["content"]),
                _extract_text(history[i + 1]["content"]),
            ))
            i += 2
        else:
            i += 1

    bfilter = None
    if book_filter and book_filter != "Tüm Kitaplar":
        bfilter = book_filter
        
    ufilter = None
    if unit_filter and unit_filter != "Tüm Birlikler":
        ufilter = unit_filter

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

def refresh_choices() -> Tuple[dict, dict, list]:
    books = rag_agent.get_ingested_books()
    book_choices = ["Tüm Kitaplar"] + books
    
    units = rag_agent.get_all_units()
    unit_choices_initial = ["Tüm Birlikler"]
    
    logger.info(f"Refreshed choices. Books: {len(books)}, Units: {len(units)}")
    
    return (
        gr.update(choices=book_choices, value="Tüm Kitaplar"), 
        gr.update(choices=unit_choices_initial, value="Tüm Birlikler"),
        units
    )

def filter_units(search_text: str, all_units: list) -> dict:
    if not all_units:
        return gr.update(choices=["Tüm Birlikler"])
        
    if not search_text:
        return gr.update(choices=["Tüm Birlikler"] + all_units[:50])
    
    search_lower = search_text.lower()
    filtered = [u for u in all_units if search_lower in u.lower()]
    limited_results = filtered[:100]
    
    return gr.update(choices=["Tüm Birlikler"] + limited_results, value="Tüm Birlikler" if not limited_results else limited_results[0])

def inspect_data(book_filter, unit_filter, limit):
    """
    Veritabanındaki paragrafları filtreleyip tablo ve indirilebilir dosya olarak döndürür.
    """
    # Normalize edilmiş birlik adının tüm varyasyonlarını al
    unit_variations = rag_agent._get_unit_variations(unit_filter)

    data = VectorDBService.browse_paragraphs(
        book_filter=book_filter,
        unit_filter=unit_variations,
        limit=int(limit)
    )
    
    if not data:
        return pd.DataFrame(columns=["Kitap", "Sayfa", "Birlikler", "Metin"]), None
        
    df = pd.DataFrame(data)
    
    # CSV dosyasını oluştur
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    
    return df, tmp.name

def build_ui() -> gr.Blocks:
    with gr.Blocks(title=settings.GRADIO_TITLE) as app:
        gr.Markdown(f"# {settings.GRADIO_TITLE}")
        
        all_units_state = gr.State([])
        
        with gr.Tabs():
            # --- SEKME 1: SOHBET ---
            with gr.Tab("💬 Asistan"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Filtreleme")
                        unit_search_input = gr.Textbox(
                            label="🔍 Birlik Ara",
                            placeholder="Örn: 3. Tümen",
                            lines=1
                        )
                        chat_unit_filter = gr.Dropdown(
                            label="🎯 Birlik Seç",
                            choices=["Tüm Birlikler"],
                            value="Tüm Birlikler",
                            interactive=True,
                            allow_custom_value=True
                        )
                        chat_book_filter = gr.Dropdown(
                            label="📚 Kitap Seç",
                            choices=["Tüm Kitaplar"],
                            value="Tüm Kitaplar",
                            interactive=True,
                            filterable=True
                        )
                        refresh_btn = gr.Button("🔄 Listeleri Yenile")

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="Sohbet", height=500)
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Sorunuzu yazın",
                                placeholder="Örn: Bu tümen hangi cephelerde savaştı?",
                                scale=4,
                            )
                            chat_btn = gr.Button("Gönder", variant="primary", scale=1)
                        
                        with gr.Accordion("Detaylar", open=False):
                            timing_box = gr.Textbox(label="Performans", interactive=False, lines=1)
                            sources_box = gr.Markdown(label="Kaynaklar")

            # --- SEKME 2: VERİ MÜFETTİŞİ ---
            with gr.Tab("🧐 Veri Müfettişi"):
                gr.Markdown("Veritabanına kaydedilen ham paragrafları buradan inceleyebilirsiniz.")
                with gr.Row():
                    with gr.Column(scale=1):
                        insp_unit_search = gr.Textbox(label="🔍 Birlik Ara", placeholder="Örn: 57. Tümen")
                        insp_unit_filter = gr.Dropdown(label="🎯 Birlik Seç", choices=["Tüm Birlikler"], value="Tüm Birlikler", allow_custom_value=True)
                        insp_book_filter = gr.Dropdown(label="📚 Kitap Seç", choices=["Tüm Kitaplar"], value="Tüm Kitaplar")
                        insp_limit = gr.Slider(label="Satır Sayısı", minimum=10, maximum=1000, value=50, step=10)
                        insp_btn = gr.Button("Verileri Getir", variant="primary")
                    
                    with gr.Column(scale=3):
                        data_table = gr.Dataframe(
                            headers=["Kitap", "Sayfa", "Birlikler", "Metin"],
                            datatype=["str", "str", "str", "str"],
                            wrap=True,
                            interactive=False
                            # height parametresi kaldırıldı
                        )
                        download_btn = gr.File(label="📥 İndir (CSV)")

            # --- SEKME 3: YÜKLEME ---
            with gr.Tab("📂 Belge Yükle"):
                pdf_input = gr.File(label="PDF Yükle", file_types=[".pdf"])
                upload_btn = gr.Button("İşle ve Veritabanına Ekle", variant="primary")
                upload_status = gr.Textbox(label="Durum", lines=10)

        # --- EVENT HANDLERS ---

        def _chat_respond(message, history, book_filter, unit_filter):
            if not message.strip(): return history, "", "", ""
            current_history = history if history else []
            answer, sources_text, timing_text = handle_chat(message, current_history, book_filter, unit_filter)
            current_history.append({"role": "user", "content": message})
            current_history.append({"role": "assistant", "content": answer})
            return current_history, "", timing_text, sources_text

        # Chat Events
        unit_search_input.change(fn=filter_units, inputs=[unit_search_input, all_units_state], outputs=[chat_unit_filter])
        chat_btn.click(fn=_chat_respond, inputs=[chat_input, chatbot, chat_book_filter, chat_unit_filter], outputs=[chatbot, chat_input, timing_box, sources_box])
        chat_input.submit(fn=_chat_respond, inputs=[chat_input, chatbot, chat_book_filter, chat_unit_filter], outputs=[chatbot, chat_input, timing_box, sources_box])
        
        # Inspector Events
        insp_unit_search.change(fn=filter_units, inputs=[insp_unit_search, all_units_state], outputs=[insp_unit_filter])
        insp_btn.click(fn=inspect_data, inputs=[insp_book_filter, insp_unit_filter, insp_limit], outputs=[data_table, download_btn])

        # Upload & Refresh Events
        upload_btn.click(fn=handle_upload, inputs=[pdf_input], outputs=[upload_status, chat_book_filter, chat_unit_filter, all_units_state])
        refresh_btn.click(fn=refresh_choices, outputs=[chat_book_filter, chat_unit_filter, all_units_state])
        
        # Init
        app.load(fn=refresh_choices, outputs=[chat_book_filter, chat_unit_filter, all_units_state])
        app.load(fn=refresh_choices, outputs=[insp_book_filter, insp_unit_filter, all_units_state])

    return app

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_port=settings.GRADIO_PORT, share=False, theme=gr.themes.Soft())
