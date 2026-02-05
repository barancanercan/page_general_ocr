import sys
from pathlib import Path

# Add the project root to the python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.main import build_ui
from src.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.GRADIO_TITLE} on port {settings.GRADIO_PORT}...")
    app = build_ui()
    app.launch(server_port=settings.GRADIO_PORT, share=False)
