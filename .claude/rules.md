# OCR Pipeline Rules

## Goal
PDF → Clean structured text. Fast, free, local.

## Hardware
- CPU: i7-11800H (16 threads)
- RAM: 64GB
- GPU: RTX 3060 6GB VRAM (CUDA)

## Stack (2026, 100% Free)
- OCR: Surya/PaddleOCR (GPU)
- LLM: Ollama (mistral/llama3/phi3)
- Embedding: local models only
- No cloud, no API costs

## Code Rules
1. No over-engineering
2. Simple, readable, minimal
3. Max 100 lines/function
4. Type hints required
5. Fail fast, clear errors

## Pipeline
```
PDF → OCR → Text Blocks → LLM Classify → Merge → Clean Paragraphs → Output
```

## Priority
Accuracy > Simplicity > Speed
