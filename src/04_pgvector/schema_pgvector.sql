-- ============================================================
-- schema_pgvector.sql
-- Module 04 — Schéma PostgreSQL + pgvector
-- Système de recommandation emploi-compétences · Cameroun
--
-- Exécution :
--   psql -U postgres -d test_kmer -f schema_pgvector.sql
-- ============================================================

-- ── EXTENSIONS ────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── TYPE ÉNUMÉRÉ : 13 types d'entités du graphe ───────────────
DO $$ BEGIN
  CREATE TYPE entity_kind AS ENUM (
    'COMPETENCE',
    'METIER',
    'OFFRE_EMPLOI',
    'CANDIDAT',
    'GROUPE_COMPETENCES',
    'GROUPE_ISCO',
    'GROUPE_BASE_MEPC',
    'SOUS_GROUPE_MEPC',
    'GRAND_GROUPE_MEPC',
    'DOMAINE_DETAILLE_NCF',
    'DOMAINE_SPECIALISE_NCF',
    'GRAND_DOMAINE_NCF',
    'NIVEAU_FORMATION_NCF'
  );
EXCEPTION
  WHEN duplicate_object THEN NULL;
END $$;

-- ── TABLE CENTRALE : embeddings ───────────────────────────────
-- Design : une seule table pour toutes les entités
-- Avantages :
--   • Requêtes ANN cross-entités en une seule requête SQL
--   • Index partiels par entity_kind pour les requêtes filtrées
--   • Gestion unifiée des mises à jour de modèle (model_id)
CREATE TABLE IF NOT EXISTS embeddings (
    id              UUID          PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_kind     entity_kind   NOT NULL,
    entity_id       TEXT          NOT NULL,   -- URI ESCO / code MEPC/NCF / UUID offre/candidat
    source_system   TEXT          NOT NULL,   -- 'ESCO' | 'MEPC' | 'NCF' | 'OFFRES' | 'CANDIDATS'
    label_fr        TEXT,                     -- Libellé principal en français
    text_to_embed   TEXT          NOT NULL,   -- Texte brut envoyé au modèle d'embedding
    embedding       VECTOR(384)   NOT NULL,   -- Vecteur dense 384d (all-MiniLM-L6-v2 fine-tuné)
    model_id        TEXT          NOT NULL
                    DEFAULT 'all-MiniLM-L6-v2-ft-offres-cm',
    neo4j_node_id   TEXT,                     -- ID du nœud Neo4j correspondant (FK logique)
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (entity_kind, entity_id, model_id)
);

-- ── INDEX HNSW PRINCIPAL ─────────────────────────────────────
-- HNSW (Hierarchical Navigable Small World) :
--   m=16 : nombre de connexions par couche (recall/mémoire)
--   ef_construction=64 : qualité de la construction (recall/temps)
--   → Recall@10 > 97% pour notre volume (~26 500 vecteurs)
--   → Latence ANN : 5-20ms
CREATE INDEX IF NOT EXISTS emb_hnsw
    ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── INDEX PARTIELS PAR TYPE (requêtes filtrées rapides) ───────
CREATE INDEX IF NOT EXISTS emb_competence_idx
    ON embeddings (entity_id)
    WHERE entity_kind = 'COMPETENCE';

CREATE INDEX IF NOT EXISTS emb_metier_idx
    ON embeddings (entity_id)
    WHERE entity_kind = 'METIER';

CREATE INDEX IF NOT EXISTS emb_offre_idx
    ON embeddings (entity_id)
    WHERE entity_kind = 'OFFRE_EMPLOI';

CREATE INDEX IF NOT EXISTS emb_candidat_idx
    ON embeddings (entity_id)
    WHERE entity_kind = 'CANDIDAT';

CREATE INDEX IF NOT EXISTS emb_ncf_det_idx
    ON embeddings (entity_id)
    WHERE entity_kind = 'DOMAINE_DETAILLE_NCF';

-- ── INDEX SUR LES COLONNES DE FILTRAGE ───────────────────────
CREATE INDEX IF NOT EXISTS emb_source_idx
    ON embeddings (source_system);

CREATE INDEX IF NOT EXISTS emb_model_idx
    ON embeddings (model_id);

CREATE INDEX IF NOT EXISTS emb_neo4j_idx
    ON embeddings (neo4j_node_id)
    WHERE neo4j_node_id IS NOT NULL;

-- ── TRIGGER : mise à jour automatique de updated_at ──────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_embeddings_updated_at ON embeddings;
CREATE TRIGGER trg_embeddings_updated_at
    BEFORE UPDATE ON embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ═════════════════════════════════════════════════════════════
-- TABLE ANALYTIQUE 1 : skill_gap
-- Stocke les écarts de compétences calculés pour chaque paire
-- (candidat, offre) — utilisée par le moteur de recommandation
-- ═════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS skill_gap (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    candidat_id     TEXT        NOT NULL,   -- ID Neo4j du candidat
    offre_id        TEXT        NOT NULL,   -- UUID de l'offre
    skill_uri       TEXT        NOT NULL,   -- URI ESCO de la compétence
    skill_label     TEXT,
    gap_type        TEXT        NOT NULL
        CHECK (gap_type IN ('ACQUISE', 'PARTIELLE', 'MANQUANTE')),
    cosine_sim      FLOAT,                  -- Score cosine pgvector (si applicable)
    neo4j_path_len  INT,                    -- Distance hiérarchique Neo4j
    source          TEXT
        CHECK (source IN ('pgvector', 'neo4j', 'llm', 'declaratif')),
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (candidat_id, offre_id, skill_uri)
);

CREATE INDEX IF NOT EXISTS sg_candidat_idx   ON skill_gap (candidat_id);
CREATE INDEX IF NOT EXISTS sg_offre_idx      ON skill_gap (offre_id);
CREATE INDEX IF NOT EXISTS sg_gap_type_idx   ON skill_gap (gap_type);
CREATE INDEX IF NOT EXISTS sg_paire_idx      ON skill_gap (candidat_id, offre_id);

-- ═════════════════════════════════════════════════════════════
-- TABLE ANALYTIQUE 2 : recommandations
-- Stocke les scores hybrides et roadmaps pour chaque paire
-- ═════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommandations (
    id               UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    candidat_id      TEXT    NOT NULL,
    offre_id         TEXT    NOT NULL,
    rang             INT,
    -- Scores composants du score hybride
    score_hybride    FLOAT   NOT NULL,   -- α·sem + β·graph + γ·collab
    score_semantique FLOAT,              -- Cosine pgvector ∈ [0,1]
    score_graph      FLOAT,              -- Score traversée Neo4j ∈ [0,1]
    score_collab     FLOAT,              -- Score collaboratif ∈ [0,1]
    -- Poids utilisés (stockés pour traçabilité)
    poids_sem        FLOAT   DEFAULT 0.40,
    poids_graph      FLOAT   DEFAULT 0.35,
    poids_collab     FLOAT   DEFAULT 0.25,
    -- Métriques skill gap
    nb_manquantes    INT,
    nb_partielles    INT,
    nb_acquises      INT,
    -- Génération LLM 2
    roadmap          JSONB,              -- Plan de formation personnalisé
    explanation      TEXT,              -- Justification textuelle
    model_version    TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (candidat_id, offre_id)
);

CREATE INDEX IF NOT EXISTS rec_score_idx     ON recommandations (score_hybride DESC);
CREATE INDEX IF NOT EXISTS rec_candidat_idx  ON recommandations (candidat_id);
CREATE INDEX IF NOT EXISTS rec_rang_idx      ON recommandations (candidat_id, rang);

-- ═════════════════════════════════════════════════════════════
-- VUES UTILITAIRES
-- ═════════════════════════════════════════════════════════════

-- Vue : statistiques par type d'entité
CREATE OR REPLACE VIEW v_embedding_stats AS
SELECT
    entity_kind,
    source_system,
    model_id,
    COUNT(*)                                    AS n_entites,
    AVG(LENGTH(text_to_embed))::INT             AS moy_chars_texte,
    MIN(created_at)                             AS premier_encode,
    MAX(updated_at)                             AS dernier_encode
FROM embeddings
GROUP BY entity_kind, source_system, model_id
ORDER BY entity_kind;

-- Vue : top recommandations par candidat
CREATE OR REPLACE VIEW v_top_recommandations AS
SELECT
    r.candidat_id,
    r.offre_id,
    r.rang,
    r.score_hybride,
    r.score_semantique,
    r.score_graph,
    r.nb_manquantes,
    r.nb_acquises,
    r.created_at
FROM recommandations r
WHERE r.rang <= 10
ORDER BY r.candidat_id, r.rang;

-- ═════════════════════════════════════════════════════════════
-- TABLE DOCUMENTAIRE : doc_chunks
-- Chunks des PDFs réglementaires indexés sémantiquement
-- Sources : NCF 2017, MEPC 2013, référentiel diplômes
-- ═════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS doc_chunks (
    id               UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id         TEXT        UNIQUE NOT NULL,   -- ex. NCF_2017_p012_c0003
    source           TEXT        NOT NULL,           -- NCF_2017 | MEPC_2013 | diplomes
    document_title   TEXT,
    page_number      INT,
    section_title    TEXT,                           -- titre du niveau hiérarchique parent
    subsection_title TEXT,
    chunk_text       TEXT        NOT NULL,
    chunk_index      INT,
    chunk_strategy   TEXT        NOT NULL
        CHECK (chunk_strategy IN ('structural', 'semantic')),
    ncf_code         TEXT,                           -- code NiveauFormationNCF / DomaineDétailléNCF
    mepc_code        TEXT,                           -- code GroupeBaseMEPC / SousGroupeMEPC
    neo4j_node_id    TEXT,                           -- ID du nœud DocChunk en Neo4j
    embedding        VECTOR(384),
    model_id         TEXT        DEFAULT 'all-MiniLM-L6-v2-ft-offres-cm',
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Index HNSW pour la recherche sémantique dans les PDFs
CREATE INDEX IF NOT EXISTS doc_chunks_hnsw
    ON doc_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS doc_chunks_source_idx ON doc_chunks (source);
CREATE INDEX IF NOT EXISTS doc_chunks_ncf_idx    ON doc_chunks (ncf_code)  WHERE ncf_code  IS NOT NULL;
CREATE INDEX IF NOT EXISTS doc_chunks_mepc_idx   ON doc_chunks (mepc_code) WHERE mepc_code IS NOT NULL;


-- ═════════════════════════════════════════════════════════════
-- REQUÊTES ANN TYPES (commentaires)
-- ═════════════════════════════════════════════════════════════

-- Top-K offres pour un candidat (ANN cosine) :
--   SELECT o.entity_id, o.label_fr,
--          1 - (c.embedding <=> o.embedding) AS cosine_sim
--   FROM   embeddings c, embeddings o
--   WHERE  c.entity_kind = 'CANDIDAT' AND c.entity_id = $candidat_id
--   AND    o.entity_kind = 'OFFRE_EMPLOI'
--   ORDER  BY c.embedding <=> o.embedding
--   LIMIT  20;

-- Compétences proches d'une compétence manquante :
--   SELECT s2.entity_id, s2.label_fr,
--          1 - (s1.embedding <=> s2.embedding) AS sim
--   FROM   embeddings s1, embeddings s2
--   WHERE  s1.entity_kind = 'COMPETENCE' AND s1.entity_id = $uri
--   AND    s2.entity_kind = 'COMPETENCE' AND s2.entity_id <> $uri
--   ORDER  BY s1.embedding <=> s2.embedding
--   LIMIT  10;
