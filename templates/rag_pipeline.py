"""
NM i AI 2026 — RAG Pipeline Template
======================================
Retrieval-Augmented Generation med sentence-transformers + ChromaDB.

Slik fungerer RAG:
1. INDEKSERING: Last inn dokumenter, del dem i biter, lag embeddings
2. SOKING: Nol bruker stiller sporsmol, finn de mest relevante bitene
3. GENERERING: Send relevante biter + sporsmol til LLM for svar

Bruk:
1. Legg dokumenter i docs/-mappen (PDF, TXT, JSON, etc.)
2. Tilpass chunking og embedding-modell
3. Kjor: python rag_pipeline.py

TODO-markerte steder ma tilpasses for den spesifikke oppgaven.
"""

import os
import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests


# === KONFIGURASJON ===

# TODO: Velg embedding-modell basert pa oppgaven
# Raske og gode norske/flersproklige modeller:
#   "all-MiniLM-L6-v2" — rask, engelsk
#   "multilingual-e5-large" — god pa norsk
#   "all-mpnet-base-v2" — beste kvalitet, engelsk
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# TODO: Velg LLM for generering
# "openai" bruker GPT-4, "anthropic" bruker Claude
LLM_PROVIDER = "anthropic"  # eller "openai"

# Chunk-innstillinger
CHUNK_SIZE = 500       # Antall tegn per chunk
CHUNK_OVERLAP = 50     # Overlapp mellom chunks (for kontekst)


# === DOKUMENTLASTING ===

def load_documents(docs_path: str) -> list[dict]:
    """
    Last inn dokumenter fra en mappe.
    Stotter: .txt, .json, .md, .csv

    TODO: Legg til stotte for PDF, HTML, etc. etter behov
    """
    documents = []
    docs_dir = Path(docs_path)

    if not docs_dir.exists():
        print(f"Mappen {docs_path} finnes ikke. Oppretter den...")
        docs_dir.mkdir(parents=True)
        return documents

    for file_path in docs_dir.rglob("*"):
        if file_path.suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
            documents.append({
                "text": text,
                "source": str(file_path),
                "type": "text",
            })

        elif file_path.suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            # TODO: Tilpass JSON-parsing til dataformatet
            if isinstance(data, list):
                for i, item in enumerate(data):
                    text = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                    documents.append({
                        "text": text,
                        "source": f"{file_path}[{i}]",
                        "type": "json",
                    })
            elif isinstance(data, dict):
                documents.append({
                    "text": json.dumps(data, ensure_ascii=False, indent=2),
                    "source": str(file_path),
                    "type": "json",
                })

        elif file_path.suffix == ".md":
            text = file_path.read_text(encoding="utf-8")
            documents.append({
                "text": text,
                "source": str(file_path),
                "type": "markdown",
            })

        elif file_path.suffix == ".csv":
            text = file_path.read_text(encoding="utf-8")
            documents.append({
                "text": text,
                "source": str(file_path),
                "type": "csv",
            })

    print(f"Lastet {len(documents)} dokumenter fra {docs_path}")
    return documents


# === CHUNKING ===

def chunk_documents(documents: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Del dokumenter i mindre biter for bedre sokeresultater.

    Hvorfor chunking?
    - Embedding-modeller har maks input-lengde
    - Mindre biter gir mer presise sok
    - Overlapp sikrer at vi ikke mister kontekst ved grenser
    """
    chunks = []

    for doc in documents:
        text = doc["text"]

        # Del teksten i chunks med overlapp
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Prov a dele pa naturlige grenser (linjeskift, punktum)
            if end < len(text):
                # Finn siste linjeskift eller punktum
                last_break = max(
                    chunk_text.rfind("\n"),
                    chunk_text.rfind(". "),
                    chunk_text.rfind("? "),
                    chunk_text.rfind("! "),
                )
                if last_break > chunk_size * 0.3:  # Minst 30% av chunk
                    chunk_text = chunk_text[:last_break + 1]
                    end = start + last_break + 1

            chunks.append({
                "text": chunk_text.strip(),
                "source": doc["source"],
                "chunk_idx": chunk_idx,
                "metadata": {
                    "source": doc["source"],
                    "type": doc["type"],
                    "chunk": chunk_idx,
                },
            })

            start = end - overlap
            chunk_idx += 1

    print(f"Delt {len(documents)} dokumenter i {len(chunks)} chunks")
    return chunks


# === VEKTORDATABASE ===

class VectorStore:
    """ChromaDB vektordatabase for effektiv likhetssok."""

    def __init__(self, collection_name: str = "nmiai_docs", persist_dir: str = "./chroma_db"):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Opprett ChromaDB-klient med lokal lagring
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )

    def add_documents(self, chunks: list[dict]):
        """Indekser chunks i vektordatabasen."""
        if not chunks:
            print("Ingen chunks a indeksere")
            return

        texts = [c["text"] for c in chunks]
        ids = [f"{c['source']}_{c['chunk_idx']}" for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Generer embeddings
        print(f"Genererer embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        # Legg til i ChromaDB
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        print(f"Indeksert {len(chunks)} chunks i vektordatabasen")

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Sok etter de mest relevante chunks for et sporsmol.

        Returnerer liste med {text, source, score}.
        """
        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "ukjent"),
                "score": 1 - results["distances"][0][i],  # Konverter distanse til likhet
            })

        return hits


# === LLM-GENERERING ===

def generate_answer(query: str, context_chunks: list[dict], provider: str = LLM_PROVIDER) -> str:
    """
    Generer svar basert pa sporsmol + relevante dokumenter.

    TODO: Tilpass system-prompten til oppgaven.
    """
    # Bygg kontekst fra sokeresultater
    context = "\n\n---\n\n".join([
        f"Kilde: {chunk['source']}\n{chunk['text']}"
        for chunk in context_chunks
    ])

    # TODO: Tilpass denne prompten til oppgaven
    system_prompt = """Du er en hjelpsom assistent som svarer pa sporsmol basert pa gitt kontekst.
Bruk KUN informasjonen i konteksten for a svare.
Hvis svaret ikke finnes i konteksten, si det klarlig.
Vær presis og konsis."""

    user_prompt = f"""Kontekst:
{context}

Sporsmol: {query}

Svar basert pa konteksten over:"""

    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt)
    elif provider == "openai":
        return _call_openai(system_prompt, user_prompt)
    else:
        # Fallback: returner bare konteksten
        return f"[Ingen LLM konfigurert]\n\nRelevant kontekst:\n{context}"


def _call_anthropic(system: str, user: str) -> str:
    """Kall Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[FEIL: Sett ANTHROPIC_API_KEY miljovariabel]"

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
    )

    data = response.json()
    return data["content"][0]["text"]


def _call_openai(system: str, user: str) -> str:
    """Kall OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "[FEIL: Sett OPENAI_API_KEY miljovariabel]"

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1024,
        },
    )

    data = response.json()
    return data["choices"][0]["message"]["content"]


# === KOMPLETT RAG-PIPELINE ===

class RAGPipeline:
    """
    Komplett RAG-pipeline: last dokumenter, indekser, sok, generer svar.
    """

    def __init__(self, docs_path: str = "./docs", collection_name: str = "nmiai_docs"):
        self.docs_path = docs_path
        self.store = VectorStore(collection_name=collection_name)
        self.indexed = False

    def index(self):
        """Last og indekser alle dokumenter."""
        documents = load_documents(self.docs_path)
        chunks = chunk_documents(documents)
        self.store.add_documents(chunks)
        self.indexed = True
        print(f"Pipeline klar — {len(chunks)} chunks indeksert")

    def query(self, question: str, n_context: int = 5) -> dict:
        """
        Still et sporsmol og fol svar.

        Returnerer:
            {
                "answer": "...",        # LLM-generert svar
                "sources": [...],       # Kilde-chunks brukt
                "query": "..."          # Opprinnelig sporsmol
            }
        """
        if not self.indexed:
            print("Indekserer dokumenter forst...")
            self.index()

        # Sok etter relevante chunks
        context_chunks = self.store.search(question, n_results=n_context)

        # Generer svar
        answer = generate_answer(question, context_chunks)

        return {
            "answer": answer,
            "sources": context_chunks,
            "query": question,
        }

    def batch_query(self, questions: list[str]) -> list[dict]:
        """Svar pa flere sporsmol."""
        results = []
        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {q}")
            result = self.query(q)
            results.append(result)
            print(f"  Svar: {result['answer'][:100]}...")
        return results


# === KJOR ===

if __name__ == "__main__":
    # TODO: Legg dokumenter i ./docs/ mappen

    # Opprett pipeline
    pipeline = RAGPipeline(docs_path="./docs")

    # Indekser dokumenter
    pipeline.index()

    # Test med noen sporsmol
    # TODO: Endre til oppgave-relevante sporsmol
    test_questions = [
        "Hva er hovedtemaet i dokumentene?",
        "Oppsummer de viktigste funnene.",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Sporsmol: {q}")
        result = pipeline.query(q)
        print(f"Svar: {result['answer']}")
        print(f"Kilder: {[s['source'] for s in result['sources'][:3]]}")
