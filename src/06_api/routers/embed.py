"""
routers/embed.py — Endpoint POST /embed
"""
import time, logging, sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from schemas import EmbedRequest, EmbedResponse
from dependencies import get_st_model

log    = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=EmbedResponse,
    summary="Encoder des textes en vecteurs 384d",
    description="""
Encode une liste de textes avec le SentenceTransformer fine-tuné
(`all-MiniLM-L6-v2-ft-offres-cm`, 384 dimensions).

Usage :
- Encoder un nouveau profil candidat avant de le stocker dans pgvector
- Encoder un intitulé de poste pour le comparer aux offres existantes
- Tester la qualité des embeddings (similarité cosine)

Les vecteurs sont normalisés (norme=1) par défaut → cosine = dot product.
""",
)
async def embed_texts(
    request: EmbedRequest,
    st_model=Depends(get_st_model),
):
    if st_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle d'embedding non disponible (vérifier le démarrage de l'API)",
        )

    t0 = time.time()
    try:
        embeddings = st_model.encode(
            request.textes,
            normalize_embeddings=request.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'encodage : {e}")

    latence_ms = round((time.time() - t0) * 1000, 1)

    return EmbedResponse(
        n_textes=len(request.textes),
        dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else 384,
        embeddings=embeddings.tolist(),
        latence_ms=latence_ms,
    )
