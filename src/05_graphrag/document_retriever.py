"""
document_retriever.py
===========================================================================
Parent-child retriever pour les chunks PDF reglementaires.

Le child retriever selectionne les chunks les plus proches. Le parent retriever
reconstruit un contexte de section/document autour du chunk pour fournir une
citation exploitable: source, page, section et extrait parent.
===========================================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DocumentEvidence:
    chunk_id: str
    source: str
    document_title: str | None
    page_number: int | None
    section_title: str | None
    subsection_title: str | None
    chunk_text: str
    parent_context: str
    cosine_sim: float
    ncf_code: str | None = None
    mepc_code: str | None = None
    neo4j_node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _embedding_literal(vec) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"


class ParentDocumentRetriever:
    """Retriever documentaire local base sur `doc_chunks`."""

    def __init__(self, conn, model, child_k: int = 8, parent_window: int = 1):
        self.conn = conn
        self.model = model
        self.child_k = child_k
        self.parent_window = parent_window

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        emb = _embedding_literal(vec)
        limit = top_k or self.child_k
        sql = """
            SELECT chunk_id, source, document_title, page_number, section_title,
                   subsection_title, chunk_text, chunk_index, ncf_code, mepc_code,
                   neo4j_node_id, 1 - (embedding <=> %s::vector) AS sim
            FROM doc_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (emb, emb, limit))
            rows = cur.fetchall()

        evidences = []
        seen = set()
        for row in rows:
            chunk_id = row[0]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            parent = self._load_parent_context(
                source=row[1],
                section_title=row[4],
                chunk_index=row[7],
            )
            evidences.append(
                DocumentEvidence(
                    chunk_id=chunk_id,
                    source=row[1],
                    document_title=row[2],
                    page_number=row[3],
                    section_title=row[4],
                    subsection_title=row[5],
                    chunk_text=str(row[6])[:900],
                    parent_context=parent,
                    ncf_code=row[8],
                    mepc_code=row[9],
                    neo4j_node_id=row[10],
                    cosine_sim=round(float(row[11]), 4),
                ).to_dict()
            )
        return evidences

    def _load_parent_context(self, *, source: str, section_title: str | None, chunk_index: int | None) -> str:
        if chunk_index is None:
            return ""
        low = max(0, int(chunk_index) - self.parent_window)
        high = int(chunk_index) + self.parent_window
        sql = """
            SELECT chunk_text
            FROM doc_chunks
            WHERE source = %s
              AND coalesce(section_title, '') = coalesce(%s, '')
              AND chunk_index BETWEEN %s AND %s
            ORDER BY chunk_index
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (source, section_title, low, high))
            rows = cur.fetchall()
        return "\n".join(str(r[0]) for r in rows)[:2000]
