"""
Interface Streamlit — Chatbot Agentic GraphRAG Emploi-Compétences Cameroun.

Lancement :
    poetry run streamlit run chatbot_app.py --server.port=8501 --server.address=127.0.0.1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
FIGURES = ROOT / "rapport" / "figures"

for _module_path in [
    SRC / "08_agentic_graphrag",
    SRC / "05_graphrag",
    SRC / "04_pgvector",
    SRC / "03_knowledge_graph",
]:
    _p = str(_module_path)
    if _p not in sys.path:
        sys.path.insert(0, _p)


st.set_page_config(
    page_title="Conseiller Emploi-Compétences — KmerAI",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS personnalisé
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Badges de timing ── */
.step-badge {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: monospace;
    margin: 2px;
    white-space: nowrap;
}
.timing-total {
    display: inline-block;
    background: #dcfce7;
    color: #166534;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-family: monospace;
    font-weight: 700;
    margin: 2px 6px 2px 0;
}

/* ── Header principal ── */
.kmerai-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    padding: 0.9rem 2rem;
    background: linear-gradient(135deg, #002d5a 0%, #00897b 100%);
    border-radius: 12px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgba(0,45,90,0.18);
}
.kmerai-header h2 {
    color: white !important;
    font-size: 1.5rem !important;
    margin: 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}
.kmerai-header p {
    color: #b2dfdb !important;
    margin: 4px 0 0 !important;
    font-size: 0.82rem !important;
}

/* ── Panneau de raisonnement ── */
div[data-testid="stStatusWidget"] {
    border-left: 3px solid #00897b !important;
    border-radius: 0 8px 8px 0 !important;
    background: #f0f9f8 !important;
}

/* ── Messages chat ── */
div[data-testid="stChatMessage"] {
    border-radius: 10px;
    margin-bottom: 0.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
LOGO_ISSEA = FIGURES / "40ansISSEA.png"
LOGO_KMERAI = FIGURES / "kmerai.png"

STEP_LABELS: dict[str, str] = {
    "analyse_request":       "🔍 Analyse de la requête",
    "plan_tools":            "🛠️  Sélection des outils",
    "execute_tools":         "⚙️  Exécution des outils",
    "build_context":         "📋 Construction du contexte",
    "generate_final_answer": "✍️  Génération de la réponse",
    "expand_context":        "🔄 Expansion du contexte",
    "critic":                "🔎 Vérification de fidélité",
}

EXAMPLES = [
    "diagnostic connexion pgvector neo4j",
    "Montre-moi les meilleures offres pour PPKOU2501080016340",
    "Analyse le skill gap du candidat PPKOU2501080016340 et propose une roadmap",
    "Je veux devenir data analyst dans une banque camerounaise",
    "Explique la classification NCF pour informatique",
    "dans le graphe quelles compétences sont liées à transit",
]


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du graphe (mis en cache)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Chargement du moteur GraphRAG…")
def load_graph():
    from graph import graph as langgraph_app  # noqa: PLC0415

    return langgraph_app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def render_trace_items(traces: list[str]) -> None:
    for trace in traces:
        st.code(str(trace), language="text")


def timing_html(timing_steps: list[dict], elapsed: float) -> str:
    """Génère les badges HTML de timing pour l'historique."""
    badges = "".join(
        f'<span class="step-badge">{s["label"]} &nbsp;{s["dt"]:.2f}s</span>'
        for s in timing_steps
    )
    return f'<span class="timing-total">⏱ {elapsed:.1f}s</span>{badges}'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logos
    col_l, col_r = st.columns(2)
    with col_l:
        if LOGO_ISSEA.exists():
            st.image(str(LOGO_ISSEA), use_container_width=True)
    with col_r:
        if LOGO_KMERAI.exists():
            st.image(str(LOGO_KMERAI), use_container_width=True)

    st.divider()
    st.markdown("### ⚙️ Paramètres")

    candidat_id_input = st.text_input(
        "ID candidat (optionnel)",
        placeholder="ex : PPKOU2501080016340",
        help="Laissez vide pour une question d'orientation générale.",
    )
    top_k = st.slider("Nombre de résultats", 1, 20, 5)
    show_traces = st.toggle("Afficher les traces du workflow", value=False)

    st.divider()
    st.markdown("**💬 Exemples de questions**")
    for idx, example in enumerate(EXAMPLES):
        if st.button(example, use_container_width=True, key=f"ex_{idx}"):
            st.session_state["pending_input"] = example
            st.rerun()

    st.divider()
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("pending_input", None)
        st.rerun()

    st.caption("GraphRAG · pgvector · Neo4j · LangGraph · KmerAI")


# ─────────────────────────────────────────────────────────────────────────────
# Header principal avec logos
# ─────────────────────────────────────────────────────────────────────────────
h_left, h_mid, h_right = st.columns([1, 5, 1])
with h_left:
    if LOGO_ISSEA.exists():
        st.image(str(LOGO_ISSEA), width=85)
with h_mid:
    st.markdown(
        "<div style='text-align:center; padding:0.4rem 0'>"
        "<h2 style='margin:0;color:#002d5a'>🇨🇲 Conseiller Emploi-Compétences</h2>"
        "<p style='margin:4px 0 0;color:#475569;font-size:0.82rem'>"
        "Système hybride · GraphRAG · pgvector · Neo4j · LangGraph · KmerAI"
        "</p></div>",
        unsafe_allow_html=True,
    )
with h_right:
    if LOGO_KMERAI.exists():
        st.image(str(LOGO_KMERAI), width=85)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Historique des messages
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Badges de timing des étapes passées
        if msg.get("timing") and msg.get("elapsed") is not None:
            st.markdown(
                timing_html(msg["timing"], msg["elapsed"]),
                unsafe_allow_html=True,
            )

        if show_traces and msg.get("traces"):
            with st.expander("Traces du workflow", expanded=False):
                render_trace_items(msg["traces"])
        if show_traces and msg.get("critic"):
            with st.expander("Critique de fidélité", expanded=False):
                st.json(msg["critic"])


# ─────────────────────────────────────────────────────────────────────────────
# Zone de saisie
# ─────────────────────────────────────────────────────────────────────────────
pending_input: str = st.session_state.pop("pending_input", "")
typed_input = st.chat_input(
    "Posez votre question… ex : Montre-moi les offres pour PPKOU2501080016340"
)
user_input: str = typed_input or pending_input

if not user_input:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Traitement de la question
# ─────────────────────────────────────────────────────────────────────────────
with st.chat_message("user"):
    st.markdown(user_input)
st.session_state["messages"].append({"role": "user", "content": user_input})

answer = ""
traces: list[str] = []
critic: dict = {}
timing_steps: list[dict] = []
elapsed = 0.0

with st.chat_message("assistant"):
    try:
        graph = load_graph()
        agent_input: dict = {
            "messages": [("user", user_input)],
            "top_k": top_k,
        }
        if candidat_id_input.strip():
            agent_input["candidat_id"] = candidat_id_input.strip()

        # ── Raisonnement pas-à-pas en surbrillance ────────────────────────────
        t0 = time.time()
        t_prev = t0
        last_state = None

        with st.status("🧠 Raisonnement en cours…", expanded=True) as status_box:
            for step in graph.stream(agent_input):
                node_name = list(step.keys())[0]
                state = list(step.values())[0]

                # Ignorer le nœud sentinelle de fin
                if node_name == "__end__":
                    last_state = state
                    continue

                t_now = time.time()
                dt_step = t_now - t_prev
                t_prev = t_now

                label = STEP_LABELS.get(node_name, f"⚡ {node_name}")

                # Détail : dernière trace de l'état courant
                step_traces = state.get("traces", [])
                last_trace = step_traces[-1] if step_traces else ""
                detail = f"— `{last_trace}`" if last_trace else ""

                # Ligne affichée dans le panneau de statut
                st.write(f"**{label}** {detail} &nbsp; `+{dt_step:.2f}s`")

                timing_steps.append({"label": label, "dt": dt_step})
                last_state = state

            elapsed = round(time.time() - t0, 1)
            status_box.update(
                label=f"✅ Terminé en **{elapsed}s**",
                state="complete",
                expanded=False,
            )

        # ── Extraction de la réponse finale ──────────────────────────────────
        if last_state:
            for message in reversed(last_state.get("messages", [])):
                if (
                    hasattr(message, "content")
                    and message.content
                    and getattr(message, "type", "") == "ai"
                ):
                    answer = str(message.content)
                    break
            traces = [str(t) for t in last_state.get("traces", [])]
            critic = last_state.get("critic", {}) or {}

        if not answer:
            answer = "Aucune réponse générée."

        # ── Affichage de la réponse ───────────────────────────────────────────
        st.markdown(answer)

        # Badges de timing sous la réponse
        st.markdown(timing_html(timing_steps, elapsed), unsafe_allow_html=True)

        if show_traces and traces:
            with st.expander("Traces du workflow", expanded=False):
                render_trace_items(traces)
        if show_traces and critic:
            with st.expander("Critique de fidélité", expanded=False):
                st.json(critic)

    except Exception as exc:
        answer = f"Erreur : `{exc}`"
        st.error(answer)

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde dans l'historique
# ─────────────────────────────────────────────────────────────────────────────
st.session_state["messages"].append(
    {
        "role": "assistant",
        "content": answer,
        "traces": traces,
        "critic": critic,
        "timing": timing_steps,
        "elapsed": elapsed,
    }
)
