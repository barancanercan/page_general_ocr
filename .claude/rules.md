# Military Document Intelligence System
## Production Rules & Architecture V3

---

## 1. Pipeline Architecture (STRICT)

```
PDF
→ OCR (word + bounding box)
→ Text Blocks (layout aware)
→ LLM Classify (TITLE / PARAGRAPH / PAGE_NUMBER / FOOTER / JUNK)
→ Block Merge (consecutive PARAGRAPHs)
→ True Paragraph
→ Unit Extraction (post-merge)
→ Embedding
→ Dual Database
```

Each stage is independent and cacheable.

---

## 2. Hardware Constraints

- CPU-only, no GPU
- Models ≤ 7B parameters
- Ollama for LLM/embedding
- Crash-free operations

---

## 3. Block-Based Paragraph Formation (CRITICAL)

The system must NOT split paragraphs based on line breaks.

OCR text must first be grouped into BLOCKS.

A BLOCK is:
- A visually contiguous region of text
- Usually separated by vertical spacing
- Usually shares font size and indentation

Each block must be classified as:
- TITLE
- HEADING
- PARAGRAPH
- PAGE_NUMBER
- FOOTER
- METADATA
- JUNK

Only blocks labeled PARAGRAPH are used.

Multiple consecutive PARAGRAPH blocks MUST be merged into ONE logical paragraph.

---

## 4. Real Paragraph Definition (UPDATED)

A real paragraph is:
- At least 200 characters OR
- At least 2 complete sentences (finite verbs)

It does NOT need punctuation to be perfect.
It does NOT require dots.

The LLM must judge semantic continuity, not typography.

---

## 5. Page Number Detection (BLOCK-BASED)

Page number detection must operate on BLOCKS, not raw text.

A block is PAGE_NUMBER if:
- It contains only digits (or — N — format)
- It is near top or bottom of page
- It is isolated from other text

Only then source_page is set.

Valid: `14`, `— 1 —`, `- 23 -`
Invalid: `Sayfa 14`, `14.`, `XIV`, `1999`

If not found: `source_page = "unknown"`

---

## 6. Military Unit Extraction (CRITICAL)

Division extraction must occur AFTER final paragraphs are formed.

Rules:
- Only extract units EXPLICITLY mentioned in text
- Use regex pre-filter + LLM verification
- NO hallucination allowed
- NO context guessing

Allowed units:
- Tümen (Division)
- Kolordu (Corps)
- Tugay (Brigade)
- Alay (Regiment)
- Fırka (historical)

Normalization:
- `24. Piyade Tümenii` → `24. Piyade Tümeni`
- `9 ncu Piyade Tümen` → `9. Piyade Tümeni`
- `Dördüncü Fırka` → `4. Tümen`

If none found: `"division": []`

---

## 7. LLM Output Rules (STRICT)

LLM must NEVER output:
- "İşte düzeltilmiş metin:"
- "Düzeltilmiş metin:"
- "Yeni bilgi:"
- Explanations or comments
- Bullet points or lists
- Any prefix/suffix text

LLM must ONLY output the cleaned text itself.

---

## 8. Confidence Model

Three metrics:
```
ocr_confidence: OCR quality (0.0-1.0)
para_quality: Paragraph reconstruction success (0.0-1.0)
entity_certainty: Entity recognition certainty (0.0-1.0)
```

Final confidence:
```
confidence = 0.4 * ocr_confidence + 0.3 * para_quality + 0.3 * entity_certainty
```

---

## 9. Canonical Paragraph Object

```json
{
  "id": "<bookid>_p<page>_para<index>",
  "document": "<clean text>",
  "embedding": [<float array>],
  "metadata": {
    "book_id": "<bookid>",
    "paragraph_index": <int>,
    "source_page": <int or "unknown">,
    "division": [<normalized unit names>],
    "confidence": <0.0-1.0>,
    "ocr_confidence": <0.0-1.0>,
    "para_quality": <0.0-1.0>,
    "entity_certainty": <0.0-1.0>
  }
}
```

---

## 10. Dual Vector Database

### Book DB
- All paragraphs from one book
- Citation fields (source_page, paragraph_index)

### Global DB
- All paragraphs from all books
- Division metadata for cross-book queries

---

## 11. Project Structure

```
/src
  /ocr
  /layout
  /paragraph
  /entity
  /scorer
  /embed
  /storage
/config
/data
/tests
```

No business logic in main.py, only orchestration.

---

## FINAL PRINCIPLE

**Accuracy > Citation > Reliability > Performance**

If a paragraph lies about a military unit, the entire system is compromised.
