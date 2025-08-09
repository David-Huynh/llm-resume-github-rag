## Resume Generator with RAG and LLM

This project provides an automated pipeline for generating personalized resumes tailored to specific job postings based on your specific experiences using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). It leverages code/project embeddings, transcript analysis, and job posting content to create impactful, targeted resumes.

### Features
- **Codebase Embedding:** Uses StarEnCoder to embed code files from previous projects into a FAISS index for efficient semantic search.
- **Semantic Retrieval:** Retrieves the most relevant code snippets and project experiences based on a job description or posting.
- **Transcript Integration:** Extracts and incorporates content from a transcript PDF (e.g., academic or work history).
- **Job Posting Scraping:** Scrapes job postings from URLs using Playwright for context extraction.
- **Personalized Resume Generation:** Uses an LLM (Ollama with Gwen3-4B-Instruct) to generate a resume tailored to the job, highlighting relevant experience and skills.

---

## Project Structure

- `embedding.py` — Embeds code files into a FAISS index for retrieval.
- `retrieval.py` — Retrieves relevant code snippets from the FAISS index given a query (e.g., job posting).
- `resume_gen_llm.py` — Main script: scrapes job postings, extracts transcript text, retrieves relevant code, and generates a personalized resume using an LLM.
- `embeds/` — Stores the generated FAISS index (`faiss.index`) and metadata (`metadata.txt`).
- `transcript.pdf` — Example transcript file to be used as input.

---

## Setup

### 1. Environment
Ensure you have Python 3.12.9 and the following packages installed:

- `transformers`
- `torch`
- `faiss-cpu` (or `faiss-gpu` if using CUDA)
- `tqdm`
- `fitz` (PyMuPDF)
- `ollama` (Python client)
- `playwright`

You can install dependencies with: 

```bash
pip install transformers torch faiss-cpu tqdm pymupdf ollama playwright
playwright install
```
Make sure you have Ollama installed too 

### 2. Embedding Your Codebase
Run the embedding script to index your code projects:

```bash
python embedding.py <path_to_code_projects> embeds/
```
This will create `embeds/faiss.index` and `embeds/metadata.txt`.

### 3. Generating a Resume
Run the main script with a job posting URL and a transcript PDF:

```bash
python resume_gen_llm.py --url <job_posting_url> --transcript transcript.pdf
```
The script will:
- Scrape the job posting
- Extract requirements using an LLM
- Retrieve relevant code/project snippets
- Extract transcript text
- Generate a personalized resume using the LLM

---

## Notes
- The project uses the `bigcode/starencoder` model for embeddings, FAISS for retrieval of past experiences and `gwen3-4b-instruct-ctx-8k` via Ollama for LLM tasks.
- You may need to adjust model names or paths depending on your hardware and environment.
- The pipeline is designed for AMD/Windows compatibility but should work on Linux as well.

---

## Credits
- Inspired by RAG pipelines and modern LLM-based resume generation techniques.
- Uses open-source models and libraries from HuggingFace, FAISS, and Ollama.
