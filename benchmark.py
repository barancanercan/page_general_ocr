import os
import time
import json
import torch
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium

# Model 1: olmOCR-2 (Doğruluk Odaklı)
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# Model 2: DeepSeek-OCR (Hız Odaklı)
from vllm import LLM, SamplingParams


def pdf_to_images(pdf_path, dpi=144):
    """PDF sayfalarını PIL formatına çevirir."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    images =  # Önceki hatanın düzeltildiği yer
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        bitmap = page.render(scale=dpi / 72)
        images.append(bitmap.to_pil())
    return images


def run_benchmark():
    input_dir = Path("./data")
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    overall_results =

    # --- MODEL 1 YÜKLEME (olmOCR-2) ---
    print("\n--- olmOCR-2 Yükleniyor (Transformers) ---")
    model_olm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-2-7B-1025", torch_dtype=torch.bfloat16
    ).to(device).eval()
    proc_olm = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # --- MODEL 2 YÜKLEME (DeepSeek-OCR) ---
    print("\n--- DeepSeek-OCR Yükleniyor (vLLM) ---")
    # vLLM ile yükleme hızı devasa artırır
    llm_ds = LLM(model="deepseek-ai/DeepSeek-OCR", mm_processor_cache_gb=0)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    for pdf_file in input_dir.glob("*.pdf"):
        print(f"\nİşleniyor: {pdf_file.name}")
        images = pdf_to_images(pdf_file)

        # 1. olmOCR-2 Testi (Sayfa Sayfa İşleme)
        start_olm = time.time()
        for i, img in enumerate(images):
            # Render ve Prompt hazırlığı
            img_b64 = render_pdf_to_base64png(str(pdf_file), i + 1, target_longest_image_dim=1288)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }]
            text = proc_olm.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            main_image = Image.open(BytesIO(base64.b64decode(img_b64)))
            inputs = proc_olm(text=[text], images=[main_image], padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                _ = model_olm.generate(**inputs, max_new_tokens=100)  # Örnekleme için kısa kestik
        duration_olm = time.time() - start_olm

        # 2. DeepSeek-OCR Testi (vLLM Batch İşleme)
        start_ds = time.time()
        ds_inputs = [
            {
                "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
                "multi_modal_data": {"image": img}
            } for img in images
        ]
        _ = llm_ds.generate(ds_inputs, sampling_params)
        duration_ds = time.time() - start_ds

        # Sonuçların Raporlanması
        res = {
            "file": pdf_file.name,
            "olmOCR_2_time": round(duration_olm, 2),
            "DeepSeek_OCR_time": round(duration_ds, 2),
            "DeepSeek_Speedup": round(duration_olm / duration_ds, 2)
        }
        overall_results.append(res)
        print(f"Bitti: {res}")

    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(overall_results, f, indent=4)


if __name__ == "__main__":
    run_benchmark()