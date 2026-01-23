#!/usr/bin/env python3
"""Military Document Intelligence System - Pipeline Orchestrator"""

import sys
from pipelines.process import process_pdf, save_jsonl


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python main.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = pdf_path.rsplit('.', 1)[0] + ".jsonl"

    print(f"Girdi: {pdf_path}")
    print(f"Çıktı: {output_path}\n")

    paragraphs = process_pdf(pdf_path)
    save_jsonl(paragraphs, output_path)

    print(f"\n✓ Tamamlandı: {len(paragraphs)} paragraf → {output_path}")


if __name__ == "__main__":
    main()
