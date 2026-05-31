"""
rag_engine.py — RAG Assistant for AvoRuby Cash v2.0
Combines ChromaDB retrieval with Ollama LLM generation
to produce contextual agricultural credit advice.
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Ensure config is importable
_this_dir = Path(__file__).resolve().parent
_src_dir = _this_dir.parent
_project_root = _src_dir.parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AvoRuby_Back.src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_TOP_K,
    RAG_TEMPERATURE,
)
from AvoRuby_Back.src.assistat.knowledge_base import get_or_build_vectorstore

# Logger
logger = logging.getLogger("AvoRubyRAG")


#  RAG ASSISTANT CLASS
class AvoRubyRAGAssistant:
    """
    RAG-powered agricultural credit advisor.
    
    Retrieves relevant documents from the ChromaDB knowledge base
    and generates contextual advice using the Ollama LLM.
    """

    def __init__(self, force_rebuild_kb: bool = False) -> None:
        logger.info("🌿 Initializing AvoRuby RAG Assistant (Ollama Local) ...")

        # 1. Load or build the vector store
        self.vectorstore = get_or_build_vectorstore(force_rebuild=force_rebuild_kb)

        # 2. Initialize the LLM
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=RAG_TEMPERATURE,
        )
        logger.info(f"LLM ready: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")

        # 3. Build the RAG chain
        self.rag_chain = self._build_rag_chain()
        logger.info("RAG chain built successfully.")

    def _build_rag_chain(self):
        """Build the LangChain RAG pipeline: Retriever → Prompt → LLM → Output."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": RAG_TOP_K}
        )

        prompt = ChatPromptTemplate.from_template(
            """Tu es un banquier et conseiller agricole expert du Maroc (AvoRuby Cash).
Profil de l'agriculteur :
{profil_context}

Extraits de la base de connaissances (MAMDA, FDA, Coopératives...) :
{context}

INSTRUCTIONS :
- Rédige UN SEUL paragraphe fluide (3 à 5 phrases max). Pas de liste à puces.
- Commence par annoncer la décision de crédit et le score SABC.
- Propose 2 actions concrètes (ex: installer le goutte-à-goutte avec subvention FDA, ou assurance MAMDA).
- Ton professionnel et marocain (mentionne les institutions locales).
- Réponds uniquement en Français."""
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": retriever | format_docs,
                "profil_context": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    @staticmethod
    def _format_profil(profil: dict) -> str:
        """Format the farmer profile dict into a readable string for the prompt."""
        equip = profil.get("equipements_manquants", [])
        equip_str = ", ".join(equip) if equip else "Aucun"
        return (
            f"Culture: {profil.get('culture', 'N/A')}\n"
            f"Région: {profil.get('localisation', 'N/A')}\n"
            f"Surface: {profil.get('superficie', 'N/A')} ha\n"
            f"Score SABC: {profil.get('sabc', 'N/A')}/100\n"
            f"Décision IA: {profil.get('decision', 'N/A')}\n"
            f"Équipements manquants: {equip_str}"
        )

    @staticmethod
    def _fallback_advice(profil: dict) -> str:
        """Fallback advice when the LLM is unavailable."""
        score = profil.get("sabc", 50)
        decision = profil.get("decision", "CONDITIONNEL")
        return (
            f"Décision : {decision}. Votre Score SABC est de {score}/100. "
            f"Contactez votre conseiller AvoRuby Cash pour optimiser votre dossier "
            f"avec la garantie CCG Damane Filaha et l'assurance MAMDA."
        )

    def get_advice(self, profil_agriculteur: dict) -> str:
        """
        Generate personalized credit advice for a farmer.
        
        Args:
            profil_agriculteur: Dict with keys like culture, localisation,
                                sabc, decision, equipements_manquants.
        
        Returns:
            A string with the RAG-generated advice, or fallback text on error.
        """
        try:
            profil_text = self._format_profil(profil_agriculteur)
            logger.info("Sending query to RAG chain ...")
            response = self.rag_chain.invoke(profil_text)
            return response.strip()
        except Exception as exc:
            logger.error(f"RAG error, using fallback: {exc}")
            return self._fallback_advice(profil_agriculteur)


#  SINGLETON ACCESS — Used by models.py and app.py
_rag_singleton: AvoRubyRAGAssistant | None = None


def init_rag(force_rebuild_kb: bool = False) -> AvoRubyRAGAssistant:
    """Initialize (or return existing) RAG assistant singleton."""
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = AvoRubyRAGAssistant(force_rebuild_kb=force_rebuild_kb)
    return _rag_singleton


def get_assistant_advice(profil: dict) -> str:
    """Convenience function — initializes RAG if needed, then generates advice."""
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = AvoRubyRAGAssistant()
    return _rag_singleton.get_advice(profil)