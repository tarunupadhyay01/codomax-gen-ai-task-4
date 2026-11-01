# Chat With Your Documents (RAG)

Module 4 project of the Generative AI Virtual Internship at **Codomax Digital Solutions**.

**Author:** Tarun Upadhyay

Upload PDF or text documents and ask questions about them. The app finds the most relevant parts of your documents and gives them to the Gemini model, so answers come from your files instead of the model's memory.

## RAG pipeline
```
Document -> Chunking -> Embeddings -> Vector DB -> Retrieval -> LLM -> Answer
```

| Step | What happens | Where |
|---|---|---|
| Document | PDF/TXT/MD text is extracted page by page | `load_document()` |
| Chunking | Text is split into overlapping chunks (default 800 characters, 150 overlap) | `chunk_text()` |
| Embeddings | Each chunk becomes a vector using `gemini-embedding-001` (`RETRIEVAL_DOCUMENT`) | `embed_texts()` |
| Vector DB | Vectors are stored in a FAISS index (cosine similarity) | `VectorStore` |
| Retrieval | The question is embedded (`RETRIEVAL_QUERY`) and the top-k similar chunks are found | `retrieve()` |
| LLM | Gemini answers using only the retrieved chunks | `answer()` |
| Answer | The reply shows the source file, page and chunk text | `app.py` |

## Retrieval quality improvements
- Overlapping chunks so sentences are not cut off from their context
- Cuts at sentence ends where possible
- Different embedding task types for documents and queries
- Adjustable chunk size, overlap and top-k
- The prompt tells the model to answer only from the context and say so when the answer is missing
- Sources are shown with similarity scores so answers can be verified

## Project structure
```
.
├── app.py             # Streamlit user interface
├── rag_core.py        # Loading, chunking, embeddings, vector DB, retrieval, answer
├── requirements.txt
├── sample_document.txt
└── README.md
```

## Setup
1. Install packages:
   ```
   pip install -r requirements.txt
   ```
2. Create a file named `.env` in the project folder (never upload it to GitHub):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Upload `sample_document.txt` or your own PDF, click **Process documents**, then ask questions.

## Limitations
- Scanned PDFs (images) have no text, so they need OCR first.
- The vector index lives in memory and is rebuilt when the app restarts.
- Only the first 400 chunks are used, to protect the free API quota.

## Screenshots
Add 1-2 screenshots of the running app here.

## Key learnings
- RAG reduces hallucinations by grounding answers in retrieved text.
- Embeddings turn text into vectors so meaning can be compared, not just keywords.
- Chunk size and overlap strongly affect retrieval quality.
- Showing sources makes answers easy to verify.
