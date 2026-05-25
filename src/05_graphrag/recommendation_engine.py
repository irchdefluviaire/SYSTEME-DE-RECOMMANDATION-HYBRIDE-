"""
recommendation_engine.py
===========================================================================
Module 05 — Moteur de recommandation hybride (GraphRAG)

Orchestre le pipeline complet :
  1. Context builder (pgvector ANN + Neo4j Cypher)
  2. LLM 2 génératif (llama3.1 local via Ollama) ou simulation
  3. Sauvegarde des résultats (PostgreSQL)

Usage :
    python recommendation_engine.py --candidat PPKOU2501080016340
    python recommendation_engine.py --candidat all --top-k 5
    python recommendation_engine.py --benchmark   # test sur 10 candidats
===========================================================================
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "05_graphrag"))
load_dotenv(ROOT / ".env")

from context_builder    import GraphRAGContextBuilder
from prompt_templates   import (
    SYSTEM_RECOMMANDATION, USER_RECOMMANDATION,
    SYSTEM_SKILL_GAP,      USER_SKILL_GAP,
    SYSTEM_ROADMAP,        USER_ROADMAP,
    format_chatml, format_openai_messages,
    get_formations,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────
# LLM CALLER
# ─────────────────────────────────────────────────────────────────────────

class LLMCaller:
    """
    Appelle le LLM 2 génératif.
    Supporte : llama3.1 local via Ollama ou simulation.
    """

    def __init__(self, backend: str = "simulation"):
        """
        backend :
          "llama"      -> llama3.1 local via Ollama/OpenAI-compatible API
          "simulation" -> Reponse JSON simulee (demo sans LLM)
        """
        self.backend = backend
        self._model  = None
        self._pipe   = None
        self._client = None
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "ollama")
        self.model = os.getenv("LLM_CHOICE", "llama3.1:latest")
        self.timeout_s = int(os.getenv("LLM_TIMEOUT_S", "180"))

    def _load_mistral(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        import torch

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        self._pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=1024, temperature=0.1, do_sample=False,
        )
        log.info("Mistral-7B-Instruct chargé (4-bit NF4)")

    def _load_openai(self):
        import openai
        self._client = openai.OpenAI()
        log.info("Client OpenAI initialisé")

    def generate(self, system: str, user: str) -> str:
        """Génère une réponse JSON à partir du system + user prompt."""

        if self.backend == "llama":
            return self._call_llama(system, user)

        if self.backend in {"mistral", "openai"}:
            raise ValueError("Backend retire: utiliser 'simulation' ou 'llama'.")

        if self.backend == "mistral":
            if self._pipe is None:
                self._load_mistral()
            prompt = format_chatml(system, user)
            output = self._pipe(prompt)[0]["generated_text"]
            # Extraire la partie après [/INST]
            response = output.split("[/INST]")[-1].strip()
            return self._extract_json(response)

        elif self.backend == "openai":
            if self._client is None:
                self._load_openai()
            messages = format_openai_messages(system, user)
            completion = self._client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            return completion.choices[0].message.content

        elif self.backend == "simulation":
            return self._simulate_response(user)

        raise ValueError(f"Backend LLM inconnu: {self.backend!r}")

    def _call_llama(self, system: str, user: str) -> str:
        """Appelle llama3.1 via Ollama en mode OpenAI-compatible."""
        payload = {
            "model": self.model,
            "messages": format_openai_messages(system, user),
            "temperature": 0.1,
            "max_tokens": 2000,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Backend llama indisponible sur {self.base_url}. "
                "Verifier que Ollama est lance et que llama3.1:latest est installe."
            ) from exc

        content = body["choices"][0]["message"]["content"]
        return self._extract_json(content)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extrait le JSON d'une réponse LLM (peut contenir du texte parasite)."""
        # Chercher le premier '{' et le dernier '}'
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text

    @staticmethod
    def _simulate_response(user_text: str) -> str:
        """Génère une réponse JSON simulée réaliste."""
        import random
        rng = random.Random(hash(user_text[:50]) % 10000)

        if "roadmap" in user_text.lower() or "étapes" in user_text.lower():
            return json.dumps({
                "poste_cible": "Poste extrait du contexte",
                "score_matching_actuel": round(rng.uniform(0.35, 0.65), 2),
                "score_matching_projete": round(rng.uniform(0.70, 0.90), 2),
                "duree_totale_estimee": f"{rng.randint(3, 12)} mois",
                "etapes": [
                    {
                        "priorite": 1,
                        "competence_cible": "Compétence identifiée par le système",
                        "type": "technique",
                        "importance": "essentielle",
                        "formation": {
                            "nom": "Formation recommandée",
                            "etablissement": "MOOC AUF ou IUT local",
                            "duree": f"{rng.randint(1, 4)} mois",
                            "cout_estimatif": "Gratuit à 50 000 FCFA",
                            "modalite": "en ligne",
                            "lien_info": "https://mooc.auf.org",
                        },
                        "impact_score": round(rng.uniform(0.05, 0.15), 2),
                        "delai_acquisition": "2-3 mois",
                    }
                ],
                "ressources_gratuites": [
                    "MOOC AUF (mooc.auf.org)",
                    "Coursera avec aide financière",
                    "YouTube EDU en français",
                ],
                "certifications_utiles": ["Certification Google Digital Skills for Africa"],
                "conseil_candidature_immediate": (
                    "Postulez dès maintenant en mettant en avant vos compétences "
                    "actuelles et votre plan de formation."
                ),
                "message_motivation": (
                    "Votre profil est prometteur. Avec quelques mois de formation ciblée, "
                    "vous serez pleinement qualifié pour ce poste."
                ),
            }, ensure_ascii=False, indent=2)

        elif "skill_gap" in user_text.lower() or "manquantes" in user_text.lower():
            taux = round(rng.uniform(0.35, 0.75), 2)
            return json.dumps({
                "taux_matching": taux,
                "niveau_gap": "modéré" if taux > 0.5 else "important",
                "competences_critiques": [
                    {
                        "label": "Compétence technique identifiée",
                        "importance": "essentielle",
                        "impact_score": 0.12,
                        "formation_recommandee": "Formation spécialisée disponible au Cameroun",
                        "delai_acquisition": "2-3 mois",
                        "ressource": "MOOC AUF ou formation professionnelle locale",
                    }
                ],
                "competences_acquises_valeur": (
                    "Le candidat dispose de compétences de base solides "
                    "qui facilitent l'apprentissage des compétences manquantes."
                ),
                "score_projete_apres_formation": round(min(taux + 0.25, 0.95), 2),
                "eligible_maintenant": taux >= 0.55,
                "message_candidat": (
                    "Votre profil présente un potentiel réel pour ce poste. "
                    "Un plan de formation ciblé vous permettra d'atteindre le niveau requis."
                ),
            }, ensure_ascii=False, indent=2)

        else:  # recommandation
            score = round(rng.uniform(0.55, 0.82), 3)
            return json.dumps({
                "analyse_globale": (
                    "Le candidat présente un profil compatible avec plusieurs offres "
                    "du marché camerounais. Les recommandations ci-dessous sont classées "
                    "par score de matching hybride."
                ),
                "score_employabilite_global": score,
                "recommandations": [
                    {
                        "rang": 1,
                        "offre_id": "offre-simulée-001",
                        "titre_poste": "Poste recommandé par le système",
                        "score_hybride": score,
                        "pourquoi_recommandee": (
                            "Cette offre correspond bien à votre profil "
                            "en termes de secteur et de niveau d'études requis."
                        ),
                        "points_forts": [
                            "Niveau d'études compatible",
                            "Secteur aligné avec votre expérience",
                        ],
                        "points_attention": [
                            "Quelques compétences techniques à renforcer",
                        ],
                        "verdict": "Recommandation forte" if score > 0.7 else "Recommandation modérée",
                    }
                ],
                "conseil_global": (
                    "Mettez en avant vos atouts lors de la candidature et "
                    "préparez-vous à démontrer votre capacité d'adaptation rapide."
                ),
                "prochaine_action": (
                    "Postulez en ligne cette semaine et préparez un CV adapté "
                    "au secteur de l'offre."
                ),
            }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────
# MOTEUR DE RECOMMANDATION
# ─────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Moteur principal de recommandation hybride (GraphRAG).

    Pipeline :
      1. Charger profil candidat
      2. Construire contexte (ANN pgvector + Cypher Neo4j)
      3. Invoquer LLM 2 avec le contexte
      4. Parser + valider la réponse JSON
      5. Générer skill gap + roadmap pour le top-1
      6. Sauvegarder dans PostgreSQL
    """

    def __init__(
        self,
        neo4j_driver=None,
        pg_conn=None,
        st_model=None,
        llm_backend: str = "simulation",
        top_k: int = 5,
    ):
        self.builder = GraphRAGContextBuilder(
            neo4j_driver=neo4j_driver,
            pg_conn=pg_conn,
            st_model=st_model,
            top_k_pgvector=20,
            top_k_final=top_k,
        )
        self.llm    = LLMCaller(backend=llm_backend)
        self.pg     = pg_conn
        self.top_k  = top_k

    def _load_candidat(self, candidat_id: str) -> dict:
        """Charge le profil d'un candidat depuis le Parquet."""
        df = pd.read_parquet(ROOT / "data" / "processed" / "candidats_normalized.parquet")
        row = df[df["candidat_id"].astype(str) == str(candidat_id)]
        if row.empty:
            log.warning(f"Candidat {candidat_id} non trouvé, utilisation du premier")
            row = df.iloc[[0]]
        r = row.iloc[0]
        return {
            "candidat_id":       str(r["candidat_id"]),
            "metier_vise":       str(r.get("metier_vise", "") or ""),
            "secteur_metier":    str(r.get("secteur_metier", "") or ""),
            "ncf_niveau_final":  int(r["ncf_niveau_final"]) if pd.notna(r.get("ncf_niveau_final")) else None,
            "filiere_specialite":str(r.get("filiere_specialite", "") or ""),
            "objectif":          str(r.get("objectif", "") or "")[:200],
            "diplome_raw":       str(r.get("diplome_raw", "") or ""),
            "secteur_demande":   str(r.get("secteur_demande", "") or ""),
            "mobilite_geo_bool": r.get("mobilite_geo_bool"),
        }

    def _safe_parse_json(self, text: str) -> Optional[dict]:
        """Parse JSON avec fallback gracieux."""
        if not text:
            return None
        clean = re.sub(r"```(?:json)?|```", "", text).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Tentative de récupération : trouver le premier bloc JSON
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(clean[start:end])
                except:
                    pass
        log.warning("JSON invalide dans la réponse LLM — réponse brute conservée")
        return {"raw_response": text[:500]}

    def recommend(self, candidat_id: str) -> dict:
        """
        Pipeline complet pour un candidat.
        Retourne le résultat structuré : offres + skill_gap + roadmap.
        """
        t0 = time.time()
        log.info(f"=== Recommandation pour {candidat_id} ===")

        # 1. Charger profil
        candidat = self._load_candidat(candidat_id)
        log.info(f"  Profil : {candidat['metier_vise']} / {candidat['secteur_metier']}")

        # 2. Context GraphRAG
        ctx = self.builder.build_context(candidat_id, candidat)
        log.info(f"  Contexte : {ctx['n_candidats']} candidats ANN → top {len(ctx['top_offres'])} offres")

        # 3. LLM 2 → Recommandations
        user_rec = USER_RECOMMANDATION.format(context_text=ctx["context_text"])
        raw_rec  = self.llm.generate(SYSTEM_RECOMMANDATION, user_rec)
        rec_json = self._safe_parse_json(raw_rec) or {}
        log.info(f"  LLM recommandation : {len(rec_json.get('recommandations', []))} offres")

        # 4. LLM 2 → Skill Gap (sur la top-1 offre)
        top1 = ctx["top_offres"][0] if ctx["top_offres"] else {}
        sg_json = {}
        if top1:
            user_sg = USER_SKILL_GAP.format(
                candidat_id=candidat_id,
                metier_vise=candidat["metier_vise"],
                ncf_niveau=candidat.get("ncf_niveau_final", "?"),
                titre_offre=top1.get("titre", ""),
                secteur=top1.get("secteur", ""),
                n_acquises=len(top1.get("acquises", [])),
                n_manquantes=len(top1.get("manquantes", [])),
                acquises_list="\n".join(f"  - {c}" for c in top1.get("acquises", [])[:5]),
                manquantes_list="\n".join(f"  - {c}" for c in top1.get("manquantes", [])[:5]),
                taux_match=top1.get("taux_match", 0.5),
            )
            raw_sg  = self.llm.generate(SYSTEM_SKILL_GAP, user_sg)
            sg_json = self._safe_parse_json(raw_sg) or {}

        # 5. LLM 2 → Roadmap (sur la top-1 offre)
        rm_json = {}
        if top1 and sg_json:
            score_actuel  = top1.get("score_hybride", 0.5)
            score_projete = sg_json.get("score_projete_apres_formation",
                                        min(score_actuel + 0.2, 0.95))
            formations    = get_formations(top1.get("secteur", ""),
                                           str(top1.get("ess_manq", "")))
            user_rm = USER_ROADMAP.format(
                candidat_profile=json.dumps(candidat, ensure_ascii=False, indent=2)[:800],
                offre_profile=json.dumps({
                    "titre":   top1.get("titre", ""),
                    "secteur": top1.get("secteur", ""),
                    "ville":   top1.get("ville", ""),
                }, ensure_ascii=False),
                competences_manquantes="\n".join(
                    f"  - {c}" for c in top1.get("manquantes", [])[:6]
                ),
                score_actuel=score_actuel,
                score_projete=score_projete,
            )
            raw_rm  = self.llm.generate(SYSTEM_ROADMAP, user_rm)
            rm_json = self._safe_parse_json(raw_rm) or {}

        elapsed = time.time() - t0
        log.info(f"  Pipeline terminé en {elapsed:.2f}s")

        result = {
            "candidat_id":     candidat_id,
            "candidat":        candidat,
            "top_offres":      ctx["top_offres"],
            "recommandations": rec_json,
            "skill_gap":       sg_json,
            "roadmap":         rm_json,
            "elapsed_s":       round(elapsed, 2),
            "n_offres_ann":    ctx["n_candidats"],
        }

        # 6. Sauvegarde PostgreSQL (si dispo)
        if self.pg:
            self._save_to_pg(result)

        return result

    def _save_to_pg(self, result: dict):
        """Sauvegarde les résultats dans la table recommandations."""
        sql = """
        INSERT INTO recommandations
          (candidat_id, offre_id, rang, score_hybride,
           score_semantique, score_graphe, score_collab,
           nb_acquises, nb_manquantes, nb_essentielles_manquantes,
           roadmap, explanation, model_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (candidat_id, offre_id) DO UPDATE SET
          score_hybride = EXCLUDED.score_hybride,
          roadmap       = EXCLUDED.roadmap,
          explanation   = EXCLUDED.explanation
        """
        cid = result["candidat_id"]
        rec = result.get("recommandations", {}).get("recommandations", [])
        rm  = json.dumps(result.get("roadmap", {}), ensure_ascii=False)
        expl = result.get("recommandations", {}).get("conseil_global", "")

        rows = []
        for i, offre in enumerate(result["top_offres"][:5], 1):
            rows.append((
                cid, offre["offre_id"], i,
                offre.get("score_hybride", 0),
                offre.get("score_sem", 0),
                offre.get("taux_match", 0),
                offre.get("score_collab", 0.5),
                len(offre.get("acquises", [])),
                len(offre.get("manquantes", [])),
                len(offre.get("ess_manq", [])),
                rm if i == 1 else None,
                expl if i == 1 else None,
                f"all-MiniLM-L6-v2-ft | {self.llm.backend}",
            ))

        with self.pg.cursor() as cur:
            cur.executemany(sql, rows)
        self.pg.commit()
        log.info(f"  {len(rows)} recommandations sauvegardées dans PostgreSQL")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module 05 — GraphRAG Recommandation")
    parser.add_argument("--candidat",  type=str, default=None,
                        help="Matricule candidat ou 'all'")
    parser.add_argument("--backend",   type=str, default="simulation",
                        choices=["simulation", "llama"])
    parser.add_argument("--top-k",     type=int, default=5)
    parser.add_argument("--benchmark", action="store_true",
                        help="Test sur 10 candidats aléatoires")
    args = parser.parse_args()

    engine = RecommendationEngine(llm_backend=args.backend, top_k=args.top_k)

    if args.benchmark:
        df = pd.read_parquet(ROOT / "data" / "processed" / "candidats_normalized.parquet")
        sample = df.sample(min(10, len(df)), random_state=42)["candidat_id"].astype(str).tolist()
        times = []
        for cid in sample:
            t0 = time.time()
            result = engine.recommend(cid)
            elapsed = time.time() - t0
            times.append(elapsed)
            score = result["top_offres"][0]["score_hybride"] if result["top_offres"] else 0
            print(f"  {cid[:20]:<22} top1_score={score:.3f}  t={elapsed:.2f}s")
        print(f"\n  Latence moy : {sum(times)/len(times):.2f}s")
        return

    cid = args.candidat or "PPKOU2501080016340"
    result = engine.recommend(cid)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
