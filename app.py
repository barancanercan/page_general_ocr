import sys
import os
import gradio as gr

# Proje kök dizinini Python yoluna ekle
# Bu, 'src' modülünün her zaman bulunabilmesini sağlar
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.main import build_ui
from src.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.GRADIO_TITLE} on port {settings.GRADIO_PORT}...")
    
    app = build_ui()
    
    # Gradio arayüzünü başlat
    app.launch(
        server_port=settings.GRADIO_PORT, 
        share=False,
        show_error=True
    )
