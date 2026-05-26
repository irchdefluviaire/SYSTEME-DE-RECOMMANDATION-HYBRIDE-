"""
Interface Streamlit — Chatbot Agentic GraphRAG Emploi-Compétences Cameroun
Lancement : streamlit run chatbot_app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
for _p in [
    SRC / "08_agentic_graphrag",
    SRC / "05_graphrag",
    SRC / "04_pgvector",
    SRC / "03_knowledge_graph",
]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# ─── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Conseiller Emploi-Compétences",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Chargement du graph (mis en cache) ───────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du moteur GraphRAG…")
def load_graph():
    from graph import graph as lg  # noqa: PLC0415
    return lg


# ─── CSS minimal ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .st-emotion-cache-1c7y2kd { background: #f0f4ff; }
    .trace-box { font-size: 0.78rem; color: #555; background: #fafafa;
                 border-left: 3px solid #6c8ebf; padding: 8px 12px;
                 border-radius: 4px; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/cameroon.png", width=64)
    st.title("Paramètres")
    st.divider()

    candidat_id_input = st.text_input(
        "ID Candidat (optionnel)",
        placeholder="ex : PP001, CAND_042",
        help="Laissez vide pour une question d'orientation générale.",
    )

    top_k = st.slider("Nombre d'offres à analyser (top-k)", 3, 20, 5)

    show_traces = st.toggle("Afficher les traces du workflow", value=False)

    st.divider()
    st.markdown("**Exemples de questions**")
    examples = [
        "Montre-moi les meilleures offres pour PP001",
        "Quelles compétences dois-je développer pour travailler en data science dans une banque ?",
        "Analyse le profil CAND_010 et propose une roadmap",
        "Je veux devenir data analyst, que faire ?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["_prefill"] = ex

    st.divider()
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.caption("GraphRAG · pgvector · Neo4j · LangGraph")

# ─── En-tête principal ────────────────────────────────────────────────────────
st.markdown("## 🤖 Conseiller Emploi-Compétences — Cameroun")
st.caption(
    "Posez vos questions en langage naturel. "
    "Le système interroge pgvector + Neo4j et génère une réponse via Llama 3.1."
)
st.divider()

# ─── Historique ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if show_traces and msg.get("traces"):
            with st.expander("Outils appelés", expanded=False):
                for tool_name in msg["traces"]:
                    st.markdown(
                        f'<div class="trace-box">🔧 <b>{tool_name}</b></div>',
                        unsafe_allow_html=True,
                    )

# ─── Zone de saisie ───────────────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill", "")
user_input = st.chat_input(
    "Posez votre question… (ex : Montre-moi les offres pour PP001)",
    key="chat_input",
) or prefill

if user_input:
    # Affiche le message utilisateur
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Prépare l'input du workflow
    agent_input: dict = {"message": user_input, "top_k": top_k}
    if candidat_id_input.strip():
        agent_input["candidat_id"] = candidat_id_input.strip()

    # Lance le workflow LangGraph (ReAct agent)
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_Analyse en cours…_ ⏳")

        try:
            g = load_graph()
            t0 = time.time()

            # Format d'entrée ReAct agent : messages list
            agent_input_fmt = {
                "messages": [("user", user_input)],
                "top_k": top_k,
            }
            if candidat_id_input.strip():
                agent_input_fmt["candidat_id"] = candidat_id_input.strip()

            result = g.invoke(agent_input_fmt)
            elapsed = round(time.time() - t0, 1)

            # Extraire la dernière réponse AI
            messages = result.get("messages", [])
            answer = ""
            tool_calls_used = []
            for msg in reversed(messages):
                # Réponse finale de l'agent
                if hasattr(msg, "content") and msg.content and getattr(msg, "type", "") == "ai":
                    if not getattr(msg, "tool_calls", None):
                        answer = str(msg.content)
                        break
            if not answer:
                answer = "Aucune réponse générée."

            # Collecter les outils appelés pour les traces
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_used.append(tc.get("name", "?"))

            placeholder.markdown(answer)
            tools_str = " → ".join(tool_calls_used) if tool_calls_used else "aucun outil"
            st.caption(f"_{elapsed}s · outils : {tools_str}_")

            if show_traces and tool_calls_used:
                with st.expander("Outils appelés par l'agent", expanded=False):
                    for i, msg in enumerate(messages):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                st.markdown(
                                    f'<div class="trace-box">🔧 <b>{tc.get("name","?")}</b> '
                                    f'— args: {tc.get("args", {})}</div>',
                                    unsafe_allow_html=True,
                                )
                        if getattr(msg, "type", "") == "tool":
                            content_preview = str(getattr(msg, "content", ""))[:300]
                            st.markdown(
                                f'<div class="trace-box">📦 Résultat outil : {content_preview}…</div>',
                                unsafe_allow_html=True,
                            )

        except Exception as exc:
            answer = f"⚠️ Erreur : `{exc}`"
            tool_calls_used = []
            placeholder.error(answer)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "traces": tool_calls_used}
    )
