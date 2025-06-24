# Changelog

- upgraded `transformers` from **4.45.2** to **4.52.4**
- added `run.py``for unified experiment launches & result tracking
- introduced `Qwen3.py` under `embedding_model/`
- introduced `Qwen/Qwen3-Embedding-8B` under `embedding_model/__init__.py`
- adjusted `templates/rag_qa_musique` prompt to
  ```
  Conclude with “Answer: ” followed by the concise answer based solely on the given document—only the answer and no other words.
  ```
