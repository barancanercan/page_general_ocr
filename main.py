#!/usr/bin/env python3
"""PDF OCR Pipeline"""

import sys
import os
from pipelines.process import process_pdf, save_jsonl


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path> [pdf_path2 ...]")
        sys.exit(1)

    for pdf_path in sys.argv[1:]:
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue

        output = pdf_path.rsplit('.', 1)[0] + ".jsonl"
        paragraphs = process_pdf(pdf_path)

        if paragraphs:
            save_jsonl(paragraphs, output)


if __name__ == "__main__":
    main()
