import os
from pathlib import Path

# .env laden
from dotenv import load_dotenv
load_dotenv()

# 1) Einlesen/Preprocessing
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore
from typing import Optional, Any, List

# 2) Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 3) Vector Store (Postgres + pgvector)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import VectorStoreQuery

# 4) LLM (OpenAI-kompatibler Endpoint über LlamaIndex)
from llama_index.llms.openai import OpenAI as LIOpenAI

# 5) Retriever/Query Engine
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# LLMMetadata (versionssicherer Import-Pfad)
try:
    from llama_index.core.llms.types import LLMMetadata
except ImportError:
    from llama_index.core.base.llms.types import LLMMetadata


# --- Konfiguration ---
DB_NAME = os.getenv("RAG_DB_NAME", "vector_db")
DB_HOST = os.getenv("RAG_DB_HOST", "localhost")
DB_PORT = os.getenv("RAG_DB_PORT", "5432")
DB_USER = os.getenv("RAG_DB_USER", "raguser")
DB_PASS = os.getenv("RAG_DB_PASS", "ragpass")
TABLE_NAME = os.getenv("RAG_TABLE", "data_documents")

EMBED_MODEL_NAME = os.getenv("RAG_EMBED", "intfloat/multilingual-e5-base")  # multilingual

# OpenAI-kompatibler Endpoint (adesso AI Hub) – Werte ausschließlich aus .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","sk-GsX9Ax0dzOKS8gAqJaPKkQ")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL","https://adesso-ai-hub.3asabc.de/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-120b-sovereign")

DATA_PATH = Path("./data")
PDF_URL = "https://arxiv.org/pdf/2307.09288.pdf"
PDF_LOCAL = DATA_PATH / "llama2.pdf"


def ensure_data():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    if not PDF_LOCAL.exists():
        import subprocess, shlex
        cmd = f'wget --user-agent "Mozilla" "{PDF_URL}" -O "{PDF_LOCAL}"'
        subprocess.run(shlex.split(cmd), check=True)


def build_embed_model():
    return HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


# Subklasse: OpenAI-Adapter mit fester Metadata (umgeht Modellnamen-Mapping)
class OpenAIWithFixedMetadata(LIOpenAI):
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        context_window: int = 8192,
        num_output: int = 512,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,  # ggf. von der Basisklasse genutzt
            **kwargs,
        )
        self._fixed_metadata = LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            model_name=model,
        )

    @property
    def metadata(self) -> LLMMetadata:
        # Wichtig: Dieses Metadata wird vom ResponseSynthesizer/Factory genutzt
        # und verhindert die Modelnamen-Validierung.
        return self._fixed_metadata


def build_llm():
    # Base-URL in Env setzen (Kompatibilität für Clients/LlamaIndex)
    if OPENAI_BASE_URL:
        os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
        os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL

    if not OPENAI_API_KEY or not OPENAI_BASE_URL:
        raise RuntimeError("OPENAI_API_KEY oder OPENAI_BASE_URL fehlt in .env")

    llm = OpenAIWithFixedMetadata(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=512,
        context_window=8192,  # bei Bedarf auf 32768/65536 erhöhen
        num_output=512,
    )
    return llm


def connect_vector_store(embed_dim: int):
    vs = PGVectorStore.from_params(
        database=DB_NAME,
        host=DB_HOST,
        password=DB_PASS,
        port=DB_PORT,
        user=DB_USER,
        table_name=TABLE_NAME,
        embed_dim=embed_dim,
    )
    return vs


def ingest_pdf(embed_model):
    loader = PyMuPDFReader()
    documents = loader.load(file_path=str(PDF_LOCAL))

    splitter = SentenceSplitter(chunk_size=1024)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        chunks = splitter.split_text(doc.text)
        text_chunks.extend(chunks)
        doc_idxs.extend([doc_idx] * len(chunks))

    nodes = []
    for idx, text in enumerate(text_chunks):
        node = TextNode(text=text)
        node.metadata = documents[doc_idxs[idx]].metadata
        node.embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        nodes.append(node)

    return nodes


class VectorDBRetriever(BaseRetriever):
    def __init__(self, vector_store: PGVectorStore, embed_model: Any, query_mode: str = "default", similarity_top_k: int = 4) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vs_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        result = self._vector_store.query(vs_query)
        nodes_with_scores: List[NodeWithScore] = []
        for i, node in enumerate(result.nodes):
            score: Optional[float] = None
            if result.similarities is not None:
                score = result.similarities[i]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores


def main():
    print(f"TABLE_NAME: {TABLE_NAME}")
    ensure_data()

    # Embeddings + LLM
    embed_model = build_embed_model()
    llm = build_llm()

    # Embedding-Dimension ermitteln (z. B. e5-base: 768)
    test_vec = embed_model.get_text_embedding("test")
    embed_dim = len(test_vec)
    print(f"Embedding-Dimension: {embed_dim}")

    # Vector Store verbinden
    vector_store = connect_vector_store(embed_dim)

    # Ingestion (idempotent simpel – ggf. später Upsert/De-Dupe)
    nodes = ingest_pdf(embed_model)
    vector_store.add(nodes)

    # Retriever + Query Engine
    retriever = VectorDBRetriever(vector_store, embed_model, query_mode="default", similarity_top_k=4)

    # WICHTIG: Keine extra get_response_synthesizer()-Metadata-Übergabe
    # Der ResponseSynthesizer/Factory nutzt llm.metadata (von unserer Subklasse überschrieben).
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    query = "Wie schneidet Llama 2 im Vergleich zu anderen Open-Source-Modellen ab?"
    response = query_engine.query(query)

    print("\nAntwort:\n", str(response))
    if getattr(response, "source_nodes", None):
        print("\nTop-Quelle:\n", response.source_nodes[0].get_content()[:500], "...")
    else:
        print("\nKeine Quellen gefunden.")


if __name__ == "__main__":
    main()
