Aşağıda verdiğim sürüm, sizin tespit ettiğiniz **halüsinasyonlu tümen çıkarımı**, **yanlış paragraf bölme**, **yavaş mimari**, **dağınık proje yapısı** ve **RAG zayıflıkları** dahil tüm sorunları kapatacak şekilde yeniden tasarlanmış, **ürün seviyesi (production-grade)** bir `rules.md`’dir.

Bu artık bir “script projesi” değil, **askerî istihbarat motoru** tanımıdır.

---

# 📜 `rules.md`  (V2 – Production Grade Military RAG System)

```markdown
# Military Document Intelligence System
## Production Rules & Architecture

This project is a **military-grade document intelligence and retrieval system** for Turkish War of Independence sources.

It converts scanned books into:
- Verified paragraphs
- Military-unit-aware metadata
- High-precision vector databases
- RAG-based historical intelligence

This system is designed for **accuracy first, speed second, cost third**.

---

# 1. Hardware & Runtime Constraints

This system runs on a **CPU-only consumer machine**.

Rules:
- CUDA must be disabled
- No GPU assumptions
- Models ≤ 7B parameters
- Low RAM usage
- Streaming allowed
- Parallelism allowed
- Crashes are unacceptable

Pipelines must be:
- Incremental
- Restartable
- Cacheable
- Stateless between stages

---

# 2. System Architecture (Mandatory)

The system is strictly modular:

```

PDF → OCR → Block Detection → Paragraph Validation
↓
Military Entity Extraction
↓
Embedding + Metadata Builder
↓
Book Vector DB        Global Vector DB
↓
RAG Engine

```

Each stage must be:
- Independently testable
- Cacheable
- Re-runnable

No stage may call another directly.
Only data objects are passed.

---

# 3. Project Structure (Mandatory)

The codebase must follow:

```

/core
/ocr
/layout
/paragraph
/entity
/embedding
/storage
/rag

/models
/config
/pipelines
/data

````

No business logic may exist in `main.py`.
Only orchestration.

---

# 4. Canonical Paragraph Object

Every stored paragraph MUST be:

```json
{
  "id": "bookid_p0005",
  "embedding": [...],
  "document": "Full paragraph text here",
  "metadata": {
    "book_id": "istiklal_harbi_mondros",
    "paragraph_index": 5,
    "source_page": 14,
    "division": ["24. Piyade Tümeni", "9. Piyade Tümeni"],
    "confidence": 0.92
  }
}
````

Partial sentences are forbidden.
Broken lines must be merged.
Every paragraph must be standalone.

---
# 5. Paragraph Validation Layer (CRITICAL)

Before any paragraph is stored, it must pass:

A text block is REAL_PARAGRAPH only if:

* It has ≥ 1 complete sentence
* It is not a title
* It is not a list of contents
* It is not page furniture
* It is not publisher metadata
* It is not only numbers
* It is not only a heading

All blocks must be classified as:

* REAL_PARAGRAPH
* HEADING
* METADATA
* JUNK

Only REAL_PARAGRAPH enters the vector DB.

---

# 6. Page Number Detection

Printed page numbers must be detected via vision + LLM.

Rules:

* Must be numeric only
* Must be near top or bottom
* Must not be part of text

If uncertain:

```
source_page = "unknown"
```

Page numbers are NEVER embedded into paragraph text.

---

# 7. Military Unit Extraction (Hard Rule)

This is the most critical rule.

Military units must ONLY be extracted if explicitly present in text.

Allowed units:

* Tümen
* Fırka
* Tugay
* Alay
* Kolordu

LLM must normalize:

* OCR errors
* Historical spelling
* Ordinal words

Examples:

* "24. Piyade Tümenii" → "24. Piyade Tümeni"
* "Dördüncü Fırka" → "4. Tümen"

FORBIDDEN:

* Inferring a unit from context
* Adding a unit not written
* Guessing from chapter titles

If the paragraph contains no explicit unit:

```
"division": []
```

This is a hard constraint.

---

# 8. Confidence Score

Each paragraph must have:

```
confidence ∈ [0.0 – 1.0]
```

Computed from:

* OCR quality
* Paragraph completeness
* Entity extraction certainty

Low confidence means unreliable data.

---

# 9. Dual Vector Database System

Two separate vector databases must exist.

### Book DB

Contains:

* All paragraphs of one book
* Page numbers
* Divisions

Used for citation.

### Global Military DB

Contains:

* All paragraphs from all books
* All divisions

Used for:

> “Which books mention the 24. Piyade Tümeni?”

---

# 10. Performance Strategy

Speed must be achieved by:

* Caching OCR
* Caching embeddings
* Batch LLM calls
* Streaming LLM responses
* Incremental indexing

Never by reducing:

* Paragraph validation
* Entity accuracy
* Metadata quality

---

# 11. RAG Rules

LLM answers must include:

* Military unit
* Book
* Page
* Quoted paragraph

No hallucinated units allowed.

This system is not a chatbot.
It is a **military historical intelligence engine**.

---

# FINAL LAW

If a paragraph lies about a military unit,
the entire system is compromised.

Accuracy > Speed > Cost
