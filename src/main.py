import logging
import gradio as gr
from typing import List, Tuple, Optional

from src.config import settings
from src.agents.ingestion_agent import IngestionAgent
from src.agents.rag_agent import RAGAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Agents
ingestion_agent = IngestionAgent()
rag_agent = RAGAgent()

def handle_upload(pdf_file) -> Tuple[str, dict]:
    if pdf_file is None:
        return "Lutfen bir PDF dosyasi secin.", gr.update(choices=["Tum Kitaplar"], value="Tum Kitaplar")

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    
    messages = []
    def on_progress(msg: str):
        messages.append(msg)

    result = ingestion_agent.ingest_pdf(pdf_path, max_pages=0, progress_callback=on_progress)

    status_icon = {"ok": "✓", "skipped": "⚠", "error": "✗"}.get(result["status"], "?")
    summary = f"{status_icon} {result['message']}"

    final_status = "\n".join(messages) + f"\n\n{summary}"
    
    return final_status, refresh_book_choices()

def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)

def handle_chat(message: str, history: List[dict], book_filter: Optional[str] = None) -> Tuple[str, str, str]:
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
    if book_filter and book_filter != "Tum Kitaplar":
        bfilter = book_filter

    answer, sources, timing = rag_agent.chat(message, history=pair_history, book_filter=bfilter)

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

def refresh_book_choices() -> dict:
    books = rag_agent.get_ingested_books()
    choices = ["Tum Kitaplar"] + books
    logger.info(f"Refreshed book choices: {choices}")
    return gr.update(choices=choices, value="Tum Kitaplar")

def build_ui() -> gr.Blocks:
    with gr.Blocks(title=settings.GRADIO_TITLE) as app:
        gr.Markdown(f"# {settings.GRADIO_TITLE}")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Adım: Belgeleri Yükleyin")
                pdf_input = gr.File(label="PDF Yükle", file_types=[".pdf"])
                upload_btn = gr.Button("İşle ve Veritabanına Ekle", variant="primary")
                upload_status = gr.Textbox(
                    label="Yükleme Durumu", lines=8, interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 2. Adım: Soru Sorun")
                with gr.Row():
                    chat_book_filter = gr.Dropdown(
                        label="Aramayı bir kitapla sınırla (isteğe bağlı)",
                        choices=["Tum Kitaplar"],
                        value="Tum Kitaplar",
                    )
                    refresh_btn = gr.Button("🔄 Kitap Listesini Yenile")

                chatbot = gr.Chatbot(label="Sohbet", height=500)
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Sorunuzu yazın",
                        placeholder="Örn: 3. Tümen hangi cephelerde savaştı?",
                        scale=4,
                    )
                    chat_btn = gr.Button("Gönder", variant="primary", scale=1)
                
                with gr.Accordion("Detaylar", open=False):
                    timing_box = gr.Textbox(label="Performans", interactive=False, lines=1)
                    sources_box = gr.Markdown(label="Kaynaklar")

        def _chat_respond(message, history, book_filter):
            if not message.strip():
                return history, "", "", ""
            
            current_history = history if history else []
            
            answer, sources_text, timing_text = handle_chat(message, current_history, book_filter)
            
            current_history.append({"role": "user", "content": message})
            current_history.append({"role": "assistant", "content": answer})
            
            return current_history, "", timing_text, sources_text

        chat_btn.click(
            fn=_chat_respond,
            inputs=[chat_input, chatbot, chat_book_filter],
            outputs=[chatbot, chat_input, timing_box, sources_box],
        )

        chat_input.submit(
            fn=_chat_respond,
            inputs=[chat_input, chatbot, chat_book_filter],
            outputs=[chatbot, chat_input, timing_box, sources_box],
        )

        upload_btn.click(
            fn=handle_upload,
            inputs=[pdf_input],
            outputs=[upload_status, chat_book_filter],
        )

        refresh_btn.click(fn=refresh_book_choices, outputs=[chat_book_filter])
        app.load(fn=refresh_book_choices, outputs=[chat_book_filter])

    return app

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_port=settings.GRADIO_PORT, share=False, theme=gr.themes.Soft())
