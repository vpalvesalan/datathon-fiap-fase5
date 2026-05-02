"""Pipeline RAG: ingestão de PDFs → chunks → embeddings → ChromaDB → retriever.

Ingere relatórios macro (Focus, Copom, análises de corretoras) de
`data/raw/agent_docs/` e persiste embeddings em `data/processed/agent_db/`.
Usa modelo multilingual (PT-BR) para evitar perda em embeddings de texto BR.

Uso:
    # Constrói (ou recria) o vector store a partir dos PDFs:
    python -m src.agent_pipeline.rag_retriever

    # Programaticamente:
    from src.agent_pipeline.rag_retriever import get_retriever
    retriever = get_retriever()
    docs = retriever.invoke("qual a expectativa do Focus para a Selic?")
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)


def _embeddings():
    """Cria o objeto de embeddings (lazy import — evita custo no test collect)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=agent_cfg.embeddings.model_name)


def _vector_store(embeddings):
    """Cria/abre handle ChromaDB persistente."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=agent_cfg.vector_store.collection_name,
        persist_directory=agent_cfg.vector_store.persist_directory,
        embedding_function=embeddings,
    )


def ingest_pdfs(force_rebuild: bool = False) -> int:
    """Carrega todos os PDFs em data/raw/agent_docs/ no ChromaDB.

    Args:
        force_rebuild: Se True, apaga o vector store antes de reingerir.

    Returns:
        Número de chunks ingeridos.

    Raises:
        RuntimeError: Se não houver PDFs em data/raw/agent_docs/.
    """
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    persist_dir = Path(agent_cfg.vector_store.persist_directory)
    raw_dir = Path(agent_cfg.agent_docs.raw_path)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(
            f"Nenhum PDF em {raw_dir}. Adicione relatórios "
            "(Focus, Copom, análises macro) antes de rodar a ingestão."
        )

    if force_rebuild and persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir)
        logger.info("Vector store anterior removido.")

    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    for pdf in pdfs:
        logger.info("Carregando %s", pdf.name)
        docs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=agent_cfg.embeddings.chunk_size,
        chunk_overlap=agent_cfg.embeddings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name=agent_cfg.vector_store.collection_name,
        persist_directory=str(persist_dir),
    )
    logger.info(
        "Ingestão concluída: %d chunks de %d PDFs em %s",
        len(chunks), len(pdfs), persist_dir,
    )
    return len(chunks)


def get_retriever():
    """Retorna retriever pronto para uso pelo agente.

    Raises:
        RuntimeError: Se o vector store ainda não foi construído.
    """
    persist_dir = Path(agent_cfg.vector_store.persist_directory)
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        raise RuntimeError(
            f"Vector store ausente em {persist_dir}. "
            "Rode `python -m src.agent_pipeline.rag_retriever` primeiro."
        )

    vs = _vector_store(_embeddings())
    return vs.as_retriever(search_kwargs={"k": agent_cfg.rag.retrieval_k})


if __name__ == "__main__":
    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("ingest_rag")
    n = ingest_pdfs(force_rebuild=True)
    print(f"\nVector store reconstruído com {n} chunks.")
