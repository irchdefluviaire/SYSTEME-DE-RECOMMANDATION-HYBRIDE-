"""
ingest_pdfs.py
===========================================================================
Pipeline d'ingestion des PDFs réglementaires — Système emploi-compétences

Pipeline en 5 étapes :
  1. Extraction du texte (pdfplumber, avec fallback PyPDF2)
  2. Chunking structural (détection de rubriques) ou sémantique (fenêtre glissante)
  3. Embedding avec le SentenceTransformer fine-tuné (Module 02)
  4. Cache JSONL local    →  data/pdf_chunks/{source}.jsonl
  5. Insertion pgvector   →  table doc_chunks (embeddings HNSW)
     Insertion Neo4j      →  (:DocChunk)-[:EXTRAIT_DE/DEFINIT/DECRIT]→ nœuds existants

Sources supportées :
  NCF_2017   → Nomenclature Camerounaise des Formations 2017
               Liens Neo4j : NiveauFormationNCF, DomaineDétailléNCF
  MEPC_2013  → Nomenclature Camerounaise des Métiers 2013
               Liens Neo4j : GrandGroupeMEPC, SousGroupeMEPC, GroupeBaseMEPC
  diplomes   → Référentiel des diplômes et certificats
               Liens Neo4j : NiveauFormationNCF

Usage :
  python scripts/ingest_pdfs.py --pdf pdf/NCF.pdf --source NCF_2017
  python scripts/ingest_pdfs.py --pdf pdf/MEPC.pdf --source MEPC_2013 --ocr
  python scripts/ingest_pdfs.py --pdf pdf/NCF.pdf  --source NCF_2017  --dry-run
  python scripts/ingest_pdfs.py --pdf pdf/NCF.pdf  --source NCF_2017  \\
      --chunk-strategy semantic --chunk-size 800
===========================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
SRC_PG  = ROOT / "src" / "04_pgvector"
SRC_NEO = ROOT / "src" / "03_knowledge_graph"
for p in (str(SRC_PG), str(SRC_NEO)):
    if p not in sys.path:
        sys.path.insert(0, p)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Chemins ────────────────────────────────────────────────────────────────
CHUNKS_DIR  = ROOT / "data" / "pdf_chunks"
MODEL_PATH  = ROOT / "models" / "st_finetuned" / "final"
MODEL_BASE  = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID    = "all-MiniLM-L6-v2-ft-offres-cm"
DIM         = 384
MAX_CHARS   = 1500   # troncature texte avant embedding

# ── DDL doc_chunks (créée si absente) ─────────────────────────────────────
_DDL_DOC_CHUNKS = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS doc_chunks (
    id               UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id         TEXT        UNIQUE NOT NULL,
    source           TEXT        NOT NULL,
    document_title   TEXT,
    page_number      INT,
    section_title    TEXT,
    subsection_title TEXT,
    chunk_text       TEXT        NOT NULL,
    chunk_index      INT,
    chunk_strategy   TEXT        NOT NULL
        CHECK (chunk_strategy IN ('structural', 'semantic')),
    ncf_code         TEXT,
    mepc_code        TEXT,
    neo4j_node_id    TEXT,
    embedding        VECTOR(384),
    model_id         TEXT        DEFAULT 'all-MiniLM-L6-v2-ft-offres-cm',
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS doc_chunks_hnsw
    ON doc_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS doc_chunks_source_idx ON doc_chunks (source);
CREATE INDEX IF NOT EXISTS doc_chunks_ncf_idx    ON doc_chunks (ncf_code)  WHERE ncf_code  IS NOT NULL;
CREATE INDEX IF NOT EXISTS doc_chunks_mepc_idx   ON doc_chunks (mepc_code) WHERE mepc_code IS NOT NULL;
"""

# ── Titres de documents connus ────────────────────────────────────────────
_DOCUMENT_TITLES = {
    "NCF_2017":  "Nomenclature Camerounaise des Formations — 2017",
    "MEPC_2013": "Nomenclature Camerounaise des Métiers et Professions — 2013",
    "diplomes":  "Référentiel des Diplômes et Certificats — Cameroun",
}

# ── Patterns de codes réglementaires ─────────────────────────────────────
_NCF_NIVEAU_RE   = re.compile(r"\bNIVEAU\s+([1-9])\b", re.IGNORECASE)
_NCF_CODE_RE     = re.compile(r"\b(\d{1,2}(?:\.\d{1,2}){1,3})\b")
_MEPC_CODE_RE    = re.compile(r"^\s*(\d{1,3})\s+[A-ZÉÀÈÙ]", re.MULTILINE)
_HEADING_NUM_RE  = re.compile(r"^\s*(\d+(?:[.\-]\d+)*)\s*[-.:)]\s+(.+)$")
_ALL_CAPS_RE     = re.compile(r"^[A-ZÉÀÈÙÂÊÎÔÛÄËÏÖÜ\s\d\-]{8,}$")


# ─────────────────────────────────────────────────────────────────────────
# Dataclass PdfChunk
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PdfChunk:
    """Représentation d'un chunk extrait d'un PDF réglementaire."""
    chunk_id:         str
    source:           str
    document_title:   str
    page_number:      int
    section_title:    str
    subsection_title: str
    chunk_text:       str
    chunk_index:      int
    chunk_strategy:   str          # "structural" | "semantic"
    ncf_code:         str  = ""
    mepc_code:        str  = ""
    embedding:        list = field(default_factory=list)

    def text_to_embed(self) -> str:
        """Texte envoyé au modèle : titre + corps tronqué."""
        header = f"{self.section_title} {self.subsection_title}".strip()
        body   = self.chunk_text[:MAX_CHARS]
        return f"{header}\n{body}".strip()[:MAX_CHARS]

    def to_jsonl(self) -> dict:
        d = asdict(self)
        d["embedding"] = self.embedding[:8] if self.embedding else []  # preview seulement
        return d


# ─────────────────────────────────────────────────────────────────────────
# Extraction de texte PDF
# ─────────────────────────────────────────────────────────────────────────

def _extract_with_pdfplumber(pdf_path: Path) -> list[dict]:
    """
    Extrait le texte page par page avec pdfplumber.

    Returns:
        [{page: int, text: str, words: list, font_sizes: list}]
    """
    import pdfplumber

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text  = page.extract_text() or ""
            words = page.extract_words(use_text_flow=True) or []
            font_sizes = sorted(
                {round(float(w.get("size", 10)), 1) for w in words if w.get("size")},
                reverse=True,
            )
            pages.append({
                "page":       i,
                "text":       text,
                "words":      words,
                "font_sizes": font_sizes,
            })
    return pages


def _extract_with_pypdf2(pdf_path: Path) -> list[dict]:
    """Fallback d'extraction sans pdfplumber."""
    try:
        import PyPDF2
        pages = []
        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page": i, "text": text, "words": [], "font_sizes": []})
        return pages
    except ImportError:
        raise ImportError(
            "Ni pdfplumber ni PyPDF2 ne sont disponibles.\n"
            "Installer avec : poetry add pdfplumber"
        )


def extract_pdf(pdf_path: Path, use_ocr: bool = False) -> list[dict]:
    """
    Extrait le texte d'un PDF, page par page.

    Args:
        pdf_path : chemin vers le fichier PDF
        use_ocr  : forcer l'OCR pour les PDFs scannés (nécessite pytesseract)

    Returns:
        Liste de dicts {page, text, words, font_sizes}
    """
    if use_ocr:
        return _extract_with_ocr(pdf_path)

    try:
        pages = _extract_with_pdfplumber(pdf_path)
        log.info(f"  pdfplumber : {len(pages)} pages extraites")
        return pages
    except ImportError:
        log.warning("  pdfplumber absent — fallback PyPDF2")
        pages = _extract_with_pypdf2(pdf_path)
        log.info(f"  PyPDF2 : {len(pages)} pages extraites")
        return pages


def _extract_with_ocr(pdf_path: Path) -> list[dict]:
    """OCR via pdf2image + pytesseract."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise ImportError(
            "OCR nécessite : poetry add pdf2image pytesseract\n"
            "Et l'installation de Tesseract-OCR sur le système."
        )

    images = convert_from_path(str(pdf_path), dpi=200)
    pages  = []
    for i, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img, lang="fra") or ""
        pages.append({"page": i, "text": text, "words": [], "font_sizes": []})
    log.info(f"  OCR Tesseract : {len(pages)} pages")
    return pages


# ─────────────────────────────────────────────────────────────────────────
# Détection des codes réglementaires
# ─────────────────────────────────────────────────────────────────────────

def _extract_ncf_code(text: str) -> str:
    """Extrait le code NCF le plus pertinent d'un bloc de texte."""
    m = _NCF_NIVEAU_RE.search(text)
    if m:
        return m.group(1)
    # Chercher les codes de type x.y.z
    codes = _NCF_CODE_RE.findall(text[:200])
    if codes:
        # Préférer le code le plus long (plus spécifique)
        return max(codes, key=len)
    return ""


def _extract_mepc_code(text: str) -> str:
    """Extrait le code MEPC d'un bloc de texte."""
    m = _MEPC_CODE_RE.search(text[:300])
    if m:
        return m.group(1)
    return ""


# ─────────────────────────────────────────────────────────────────────────
# Chunker structural
# ─────────────────────────────────────────────────────────────────────────

class StructuralChunker:
    """
    Découpe un PDF selon sa structure hiérarchique (rubriques / sections).

    Stratégie :
      1. Détecter les lignes-titres (police ≥ médiane+2pts ou pattern numéroté)
      2. Chaque titre ouvre un nouveau chunk ; le corps suit jusqu'au titre suivant
      3. Les chunks trop courts (< 80 chars) sont fusionnés avec le précédent

    Adapté aux documents NCF et MEPC : sections numérotées ou en majuscules.
    """

    MIN_CHUNK_CHARS = 80
    MAX_CHUNK_CHARS = 3000

    def __init__(self, source: str) -> None:
        self.source = source

    def _is_heading(self, line: str, median_size: float, page_data: dict) -> bool:
        line = line.strip()
        if not line or len(line) < 3:
            return False
        if _HEADING_NUM_RE.match(line):
            return True
        if _ALL_CAPS_RE.match(line) and len(line) > 5:
            return True
        if line.upper().startswith(("NIVEAU", "GRAND GROUPE", "SOUS-GROUPE",
                                     "GROUPE DE BASE", "CHAPITRE", "PARTIE",
                                     "DOMAINE", "SECTION")):
            return True
        return False

    def chunk(
        self,
        pages:          list[dict],
        document_title: str,
    ) -> list[PdfChunk]:
        chunks: list[PdfChunk] = []
        current_title    = ""
        current_subtitle = ""
        current_body:    list[str] = []
        current_page     = 1
        idx              = 0

        def _flush():
            nonlocal idx
            body = "\n".join(current_body).strip()
            if len(body) < self.MIN_CHUNK_CHARS:
                return
            # Tronquer si trop long
            body = body[:self.MAX_CHUNK_CHARS]

            cid = f"{self.source}_p{current_page:03d}_c{idx:04d}"
            ncf  = _extract_ncf_code(current_title + " " + body) if "NCF" in self.source.upper() else ""
            mepc = _extract_mepc_code(current_title + " " + body) if "MEPC" in self.source.upper() else ""

            chunks.append(PdfChunk(
                chunk_id=cid,
                source=self.source,
                document_title=document_title,
                page_number=current_page,
                section_title=current_title[:200],
                subsection_title=current_subtitle[:200],
                chunk_text=body,
                chunk_index=idx,
                chunk_strategy="structural",
                ncf_code=ncf,
                mepc_code=mepc,
            ))
            idx += 1

        for page_data in pages:
            page_num   = page_data["page"]
            text       = page_data["text"] or ""
            font_sizes = page_data.get("font_sizes", [])
            median_sz  = float(font_sizes[len(font_sizes) // 2]) if len(font_sizes) > 2 else 10.0

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if self._is_heading(line, median_sz, page_data):
                    _flush()
                    current_body     = []
                    current_page     = page_num
                    current_subtitle = current_title
                    current_title    = line
                else:
                    current_body.append(line)

        _flush()

        # Si aucun chunk détecté (pas de structure), basculer en sémantique
        if not chunks:
            log.warning("  Structure non détectée — basculement en chunking sémantique")

        return chunks


# ─────────────────────────────────────────────────────────────────────────
# Chunker sémantique (fenêtre glissante)
# ─────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Découpe le texte en chunks de taille fixe avec overlap.

    Stratégie :
      - Découper le texte complet du document en segments de `chunk_size` chars
      - Overlap de `chunk_overlap` chars pour conserver le contexte
      - Chaque chunk porte le numéro de page d'origine approximatif
    """

    def __init__(self, source: str, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.source        = source
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        pages:          list[dict],
        document_title: str,
    ) -> list[PdfChunk]:
        # Concaténer tout le texte avec marqueurs de page
        page_breaks: list[tuple[int, int]] = []   # (char_offset, page_num)
        full_text = ""
        for pd in pages:
            page_breaks.append((len(full_text), pd["page"]))
            full_text += (pd["text"] or "") + "\n\n"

        def _page_at(offset: int) -> int:
            page = 1
            for char_off, pn in page_breaks:
                if char_off <= offset:
                    page = pn
            return page

        chunks: list[PdfChunk] = []
        step = self.chunk_size - self.chunk_overlap
        idx  = 0

        for start in range(0, len(full_text), step):
            end  = start + self.chunk_size
            body = full_text[start:end].strip()
            if len(body) < 40:
                continue

            page = _page_at(start)
            # Extraire une ligne-titre (première ligne non vide)
            title = next(
                (l.strip() for l in body.splitlines() if len(l.strip()) > 5),
                f"Chunk {idx}",
            )

            ncf  = _extract_ncf_code(body) if "NCF" in self.source.upper() else ""
            mepc = _extract_mepc_code(body) if "MEPC" in self.source.upper() else ""

            chunks.append(PdfChunk(
                chunk_id=f"{self.source}_s{start:07d}_c{idx:04d}",
                source=self.source,
                document_title=document_title,
                page_number=page,
                section_title=title[:200],
                subsection_title="",
                chunk_text=body,
                chunk_index=idx,
                chunk_strategy="semantic",
                ncf_code=ncf,
                mepc_code=mepc,
            ))
            idx += 1

        return chunks


# ─────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────

def load_model(online: bool = False):
    """Charge le SentenceTransformer fine-tuné (ou le baseline si absent)."""
    from sentence_transformers import SentenceTransformer

    if MODEL_PATH.exists():
        log.info(f"  Modèle fine-tuné : {MODEL_PATH}")
        return SentenceTransformer(str(MODEL_PATH), local_files_only=not online)

    log.warning(f"  Modèle fine-tuné absent → baseline : {MODEL_BASE}")
    return SentenceTransformer(MODEL_BASE, local_files_only=not online)


def embed_chunks(
    chunks:     list[PdfChunk],
    model,
    batch_size: int = 64,
) -> list[PdfChunk]:
    """Encode tous les chunks avec le SentenceTransformer."""
    if not chunks:
        return chunks

    texts = [c.text_to_embed() for c in chunks]
    log.info(f"  Encoding {len(texts)} chunks…")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    for chunk, vec in zip(chunks, vecs):
        chunk.embedding = vec.tolist()
    return chunks


# ─────────────────────────────────────────────────────────────────────────
# Cache JSONL
# ─────────────────────────────────────────────────────────────────────────

def write_jsonl_cache(chunks: list[PdfChunk], source: str) -> Path:
    """Écrit les chunks dans data/pdf_chunks/{source}.jsonl (sans les embeddings complets)."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHUNKS_DIR / f"{source}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            record = {
                "chunk_id":         c.chunk_id,
                "source":           c.source,
                "document_title":   c.document_title,
                "page_number":      c.page_number,
                "section_title":    c.section_title,
                "subsection_title": c.subsection_title,
                "chunk_text":       c.chunk_text[:500],   # aperçu
                "chunk_index":      c.chunk_index,
                "chunk_strategy":   c.chunk_strategy,
                "ncf_code":         c.ncf_code,
                "mepc_code":        c.mepc_code,
                "has_embedding":    bool(c.embedding),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"  Cache JSONL → {path}  ({len(chunks)} chunks)")
    return path


# ─────────────────────────────────────────────────────────────────────────
# pgvector — table doc_chunks
# ─────────────────────────────────────────────────────────────────────────

def setup_pgvector_schema(conn) -> None:
    """Crée la table doc_chunks et ses index si absents."""
    with conn.cursor() as cur:
        cur.execute(_DDL_DOC_CHUNKS)
    conn.commit()


def upsert_to_pgvector(
    chunks: list[PdfChunk],
    conn,
    batch_size: int = 200,
) -> int:
    """
    Insère ou met à jour les chunks dans pgvector (ON CONFLICT DO UPDATE).

    Returns:
        Nombre de chunks traités.
    """
    sql = """
        INSERT INTO doc_chunks
            (chunk_id, source, document_title, page_number,
             section_title, subsection_title, chunk_text,
             chunk_index, chunk_strategy, ncf_code, mepc_code,
             embedding, model_id)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
        ON CONFLICT (chunk_id) DO UPDATE SET
            chunk_text       = EXCLUDED.chunk_text,
            embedding        = EXCLUDED.embedding,
            model_id         = EXCLUDED.model_id,
            ncf_code         = EXCLUDED.ncf_code,
            mepc_code        = EXCLUDED.mepc_code
    """
    count = 0
    with conn.cursor() as cur:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            rows  = []
            for c in batch:
                if not c.embedding:
                    continue
                emb_str = "[" + ",".join(f"{v:.6f}" for v in c.embedding) + "]"
                rows.append((
                    c.chunk_id, c.source, c.document_title, c.page_number,
                    c.section_title, c.subsection_title, c.chunk_text,
                    c.chunk_index, c.chunk_strategy,
                    c.ncf_code or None, c.mepc_code or None,
                    emb_str, MODEL_ID,
                ))
            if rows:
                cur.executemany(sql, rows)
                count += len(rows)
        conn.commit()
    return count


# ─────────────────────────────────────────────────────────────────────────
# Neo4j — nœuds DocChunk et relations
# ─────────────────────────────────────────────────────────────────────────

_NEO4J_CONSTRAINTS = [
    "CREATE CONSTRAINT doc_chunk_id IF NOT EXISTS FOR (d:DocChunk) REQUIRE d.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT doc_ref_source IF NOT EXISTS FOR (r:DocumentReferentiel) REQUIRE r.source IS UNIQUE",
]

_CYPHER_DOC_REF = """
MERGE (r:DocumentReferentiel {source: $source})
  ON CREATE SET r.title = $title, r.created_at = datetime()
"""

_CYPHER_CHUNK = """
MERGE (d:DocChunk {chunk_id: $chunk_id})
  ON CREATE SET
    d.source           = $source,
    d.page_number      = $page_number,
    d.section_title    = $section_title,
    d.subsection_title = $subsection_title,
    d.chunk_text_preview = $chunk_text_preview,
    d.chunk_strategy   = $chunk_strategy,
    d.ncf_code         = $ncf_code,
    d.mepc_code        = $mepc_code,
    d.created_at       = datetime()
  ON MATCH SET
    d.chunk_text_preview = $chunk_text_preview
WITH d
MATCH (r:DocumentReferentiel {source: $source})
MERGE (d)-[:EXTRAIT_DE]->(r)
"""

_CYPHER_LINK_NCF_NIVEAU = """
MATCH (d:DocChunk {chunk_id: $chunk_id})
MATCH (n:NiveauFormationNCF {code: $ncf_code})
MERGE (d)-[:DEFINIT]->(n)
"""

_CYPHER_LINK_NCF_DET = """
MATCH (d:DocChunk {chunk_id: $chunk_id})
MATCH (n:DomaineDétailléNCF {code: $ncf_code})
MERGE (d)-[:DEFINIT]->(n)
"""

_CYPHER_LINK_MEPC_BASE = """
MATCH (d:DocChunk {chunk_id: $chunk_id})
MATCH (g:GroupeBaseMEPC {code: $mepc_code})
MERGE (d)-[:DECRIT]->(g)
"""

_CYPHER_LINK_MEPC_SOUS = """
MATCH (d:DocChunk {chunk_id: $chunk_id})
MATCH (g:SousGroupeMEPC {code: $mepc_code})
MERGE (d)-[:DECRIT]->(g)
"""

_CYPHER_LINK_MEPC_GRAND = """
MATCH (d:DocChunk {chunk_id: $chunk_id})
MATCH (g:GrandGroupeMEPC {code: $mepc_code})
MERGE (d)-[:DECRIT]->(g)
"""


def setup_neo4j_schema(driver, database: str = "neo4j") -> None:
    """Crée les contraintes DocChunk et DocumentReferentiel."""
    with driver.session(database=database) as session:
        for cypher in _NEO4J_CONSTRAINTS:
            try:
                session.run(cypher)
            except Exception as exc:
                log.debug(f"  Contrainte Neo4j (ignorée si existe) : {exc}")


def upsert_to_neo4j(
    chunks:   list[PdfChunk],
    driver,
    document_title: str,
    source:         str,
    database:       str = "neo4j",
    batch_size:     int = 100,
) -> int:
    """
    Crée les nœuds DocChunk dans Neo4j et établit les relations vers
    DocumentReferentiel, NiveauFormationNCF, DomaineDétailléNCF, GroupeBaseMEPC.

    Returns:
        Nombre de nœuds créés ou mis à jour.
    """
    count = 0
    with driver.session(database=database) as session:
        # DocumentReferentiel
        session.run(_CYPHER_DOC_REF, source=source, title=document_title)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            for c in batch:
                params = {
                    "chunk_id":           c.chunk_id,
                    "source":             c.source,
                    "page_number":        c.page_number,
                    "section_title":      c.section_title[:200],
                    "subsection_title":   c.subsection_title[:200],
                    "chunk_text_preview": c.chunk_text[:300],
                    "chunk_strategy":     c.chunk_strategy,
                    "ncf_code":           c.ncf_code or None,
                    "mepc_code":          c.mepc_code or None,
                }
                session.run(_CYPHER_CHUNK, **params)

                # Relations NCF
                if c.ncf_code:
                    if len(c.ncf_code) == 1:
                        # Niveau formation (1-9)
                        try:
                            session.run(_CYPHER_LINK_NCF_NIVEAU,
                                        chunk_id=c.chunk_id, ncf_code=c.ncf_code)
                        except Exception:
                            pass
                    else:
                        try:
                            session.run(_CYPHER_LINK_NCF_DET,
                                        chunk_id=c.chunk_id, ncf_code=c.ncf_code)
                        except Exception:
                            pass

                # Relations MEPC
                if c.mepc_code:
                    code_len = len(c.mepc_code)
                    if code_len == 1:
                        cypher = _CYPHER_LINK_MEPC_GRAND
                    elif code_len == 2:
                        cypher = _CYPHER_LINK_MEPC_SOUS
                    else:
                        cypher = _CYPHER_LINK_MEPC_BASE
                    try:
                        session.run(cypher, chunk_id=c.chunk_id, mepc_code=c.mepc_code)
                    except Exception:
                        pass

                count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────

class IngestPipeline:
    """Orchestre l'ingestion complète d'un PDF réglementaire."""

    def __init__(
        self,
        pdf_path:       Path,
        source:         str,
        chunk_strategy: str   = "structural",
        chunk_size:     int   = 800,
        chunk_overlap:  int   = 120,
        use_ocr:        bool  = False,
        dry_run:        bool  = False,
        online:         bool  = False,
        batch_size:     int   = 64,
    ) -> None:
        self.pdf_path       = pdf_path
        self.source         = source
        self.chunk_strategy = chunk_strategy
        self.chunk_size     = chunk_size
        self.chunk_overlap  = chunk_overlap
        self.use_ocr        = use_ocr
        self.dry_run        = dry_run
        self.online         = online
        self.batch_size     = batch_size
        self.doc_title      = _DOCUMENT_TITLES.get(source, source)

    def run(self) -> dict:
        """Lance le pipeline complet."""
        log.info("=" * 65)
        log.info(f"INGESTION PDF : {self.pdf_path.name}")
        log.info(f"  Source       : {self.source}")
        log.info(f"  Stratégie    : {self.chunk_strategy}")
        log.info(f"  Dry-run      : {self.dry_run}")
        log.info("=" * 65)

        t0 = time.time()
        model = None
        if not self.dry_run:
            # On Windows, importing sentence-transformers after pdfplumber can
            # crash in native pyarrow imports. Load the model before PDF parsing.
            log.info("\n[0/5] Chargement du modèle d'embedding…")
            model = load_model(self.online)

        # ── 1. Extraction ──────────────────────────────────────────────
        log.info("\n[1/5] Extraction du texte PDF…")
        pages = extract_pdf(self.pdf_path, use_ocr=self.use_ocr)
        n_chars = sum(len(p["text"]) for p in pages)
        log.info(f"  {len(pages)} pages  |  {n_chars:,} caractères")

        # ── 2. Chunking ────────────────────────────────────────────────
        log.info(f"\n[2/5] Chunking ({self.chunk_strategy})…")
        if self.chunk_strategy == "structural":
            chunker = StructuralChunker(self.source)
            chunks  = chunker.chunk(pages, self.doc_title)
            # Fallback sémantique si aucun chunk structurel
            if not chunks:
                chunker_sem = SemanticChunker(self.source, self.chunk_size, self.chunk_overlap)
                chunks      = chunker_sem.chunk(pages, self.doc_title)
        else:
            chunker = SemanticChunker(self.source, self.chunk_size, self.chunk_overlap)
            chunks  = chunker.chunk(pages, self.doc_title)

        log.info(f"  {len(chunks)} chunks générés")

        with_ncf  = sum(1 for c in chunks if c.ncf_code)
        with_mepc = sum(1 for c in chunks if c.mepc_code)
        if with_ncf  > 0: log.info(f"  Codes NCF  détectés : {with_ncf}")
        if with_mepc > 0: log.info(f"  Codes MEPC détectés : {with_mepc}")

        if not chunks:
            log.error("  Aucun chunk généré — pipeline interrompu.")
            return {"status": "error", "reason": "no_chunks"}

        # ── 3. Embedding ───────────────────────────────────────────────
        log.info("\n[3/5] Embedding des chunks…")
        if not self.dry_run:
            chunks = embed_chunks(chunks, model, self.batch_size)
        else:
            log.info("  [DRY-RUN] Embedding ignoré")
            # Vecteurs nuls pour le dry-run
            for c in chunks:
                c.embedding = [0.0] * DIM

        # ── 4. Cache JSONL ─────────────────────────────────────────────
        log.info("\n[4/5] Écriture du cache JSONL…")
        cache_path = write_jsonl_cache(chunks, self.source)

        if self.dry_run:
            elapsed = time.time() - t0
            log.info(f"\n[DRY-RUN] Pipeline terminé en {elapsed:.1f}s")
            log.info(f"  {len(chunks)} chunks  |  cache : {cache_path}")
            return {"status": "dry_run", "n_chunks": len(chunks), "cache": str(cache_path)}

        # ── 5a. pgvector ───────────────────────────────────────────────
        log.info("\n[5a/5] Insertion dans pgvector (doc_chunks)…")
        n_pg = 0
        try:
            import psycopg
            from config_pgvector import PG_CONN
            with psycopg.connect(**PG_CONN) as conn:
                setup_pgvector_schema(conn)
                n_pg = upsert_to_pgvector(chunks, conn, self.batch_size)
            log.info(f"  pgvector : {n_pg} chunks insérés/mis à jour")
        except Exception as exc:
            log.error(f"  pgvector indisponible : {exc}")

        # ── 5b. Neo4j ──────────────────────────────────────────────────
        log.info("\n[5b/5] Insertion dans Neo4j (DocChunk)…")
        n_neo = 0
        try:
            from neo4j import GraphDatabase
            from config_neo4j import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver:
                setup_neo4j_schema(driver, NEO4J_DATABASE)
                n_neo = upsert_to_neo4j(
                    chunks, driver, self.doc_title, self.source,
                    NEO4J_DATABASE, self.batch_size,
                )
            log.info(f"  Neo4j : {n_neo} nœuds DocChunk créés/mis à jour")
        except Exception as exc:
            log.error(f"  Neo4j indisponible : {exc}")

        elapsed = time.time() - t0
        log.info("\n" + "=" * 65)
        log.info(f"INGESTION TERMINÉE EN {elapsed:.1f}s")
        log.info(f"  Chunks        : {len(chunks)}")
        log.info(f"  pgvector      : {n_pg}")
        log.info(f"  Neo4j         : {n_neo}")
        log.info(f"  Cache JSONL   : {cache_path}")
        log.info("=" * 65)

        return {
            "status":   "ok",
            "n_chunks": len(chunks),
            "n_pg":     n_pg,
            "n_neo":    n_neo,
            "cache":    str(cache_path),
            "elapsed_s": round(elapsed, 1),
        }


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion des PDFs réglementaires — système emploi-compétences"
    )
    parser.add_argument(
        "--pdf", required=True, type=Path,
        help="Chemin vers le fichier PDF (ex. pdf/NCF.pdf)",
    )
    parser.add_argument(
        "--source", required=True,
        choices=["NCF_2017", "MEPC_2013", "diplomes"],
        help="Identifiant de la source réglementaire",
    )
    parser.add_argument(
        "--chunk-strategy", default="structural",
        choices=["structural", "semantic"],
        help="Stratégie de chunking (défaut : structural)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=800,
        help="Taille des chunks sémantiques en caractères (défaut : 800)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=120,
        help="Overlap entre chunks sémantiques (défaut : 120)",
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Forcer l'OCR (pdf2image + pytesseract) pour les PDFs scannés",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simuler le pipeline sans écrire dans pgvector ni Neo4j",
    )
    parser.add_argument(
        "--online", action="store_true",
        help="Autoriser le téléchargement du modèle depuis HuggingFace",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Taille des batchs d'encodage (défaut : 64)",
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path
    if not pdf_path.exists():
        log.error(f"PDF introuvable : {pdf_path}")
        sys.exit(1)

    pipeline = IngestPipeline(
        pdf_path=pdf_path,
        source=args.source,
        chunk_strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ocr=args.ocr,
        dry_run=args.dry_run,
        online=args.online,
        batch_size=args.batch_size,
    )
    result = pipeline.run()

    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
