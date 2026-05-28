"""
Interface Streamlit - Chatbot Agentic GraphRAG Emploi-Competences Cameroun.

Lancement:
    poetry run streamlit run chatbot_app.py --server.port=8501 --server.address=127.0.0.1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for module_path in [
    SRC / "08_agentic_graphrag",
    SRC / "05_graphrag",
    SRC / "04_pgvector",
    SRC / "03_knowledge_graph",
]:
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)


st.set_page_config(
    page_title="Conseiller Emploi-Competences",
    page_icon="CM",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Chargement du moteur GraphRAG...")
def load_graph():
    from graph import graph as langgraph_app  # noqa: PLC0415

    return langgraph_app


def render_trace_items(traces: list[str]) -> None:
    for trace in traces:
        st.code(str(trace), language="text")


with st.sidebar:
    st.title("Parametres")
    st.divider()

    candidat_id_input = st.text_input(
        "ID candidat (optionnel)",
        placeholder="ex : PPKOU2501080016340",
        help="Laissez vide pour une question d'orientation generale.",
    )

    top_k = st.slider("Nombre de resultats", 1, 20, 5)
    show_traces = st.toggle("Afficher les traces du workflow", value=False)

    st.divider()
    st.markdown("**Exemples de questions**")
    examples = [
        "diagnostic connexion pgvector neo4j",
        "Montre-moi les meilleures offres pour PPKOU2501080016340",
        "Analyse le skill gap du candidat PPKOU2501080016340 et propose une roadmap de formation",
        "Je veux devenir data analyst dans une banque",
        "Explique la classification NCF pour informatique",
        "dans le graphe quelles competences sont liees a transit",
    ]
    for idx, example in enumerate(examples):
        if st.button(example, use_container_width=True, key=f"example_{idx}"):
            st.session_state["pending_input"] = example
            st.rerun()

    st.divider()
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("pending_input", None)
        st.rerun()

    st.caption("GraphRAG | pgvector | Neo4j | LangGraph")


st.markdown("## Conseiller Emploi-Competences - Cameroun")
st.caption(
    "Posez une question en langage naturel. Le systeme interroge pgvector, "
    "Neo4j, les referentiels indexes et le workflow LangGraph."
)
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for stored_msg in st.session_state["messages"]:
    with st.chat_message(stored_msg["role"]):
        st.markdown(stored_msg["content"])
        if show_traces and stored_msg.get("traces"):
            with st.expander("Traces du workflow", expanded=False):
                render_trace_items(stored_msg["traces"])
        if show_traces and stored_msg.get("critic"):
            with st.expander("Critique de fidelite", expanded=False):
                st.json(stored_msg["critic"])


pending_input = st.session_state.pop("pending_input", "")
typed_input = st.chat_input(
    "Posez votre question... ex : Montre-moi les offres pour PPKOU2501080016340"
)
user_input = typed_input or pending_input

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        answer = ""
        traces: list[str] = []
        critic = {}
        try:
            with st.spinner("Analyse en cours..."):
                graph = load_graph()
                t0 = time.time()
                agent_input = {
                    "messages": [("user", user_input)],
                    "top_k": top_k,
                }
                if candidat_id_input.strip():
                    agent_input["candidat_id"] = candidat_id_input.strip()

                result = graph.invoke(agent_input)
                elapsed = round(time.time() - t0, 1)

            for message in reversed(result.get("messages", [])):
                if hasattr(message, "content") and message.content and getattr(message, "type", "") == "ai":
                    answer = str(message.content)
                    break

            if not answer:
                answer = "Aucune reponse generee."

            traces = [str(trace) for trace in result.get("traces", [])]
            critic = result.get("critic", {}) or {}

            st.markdown(answer)
            trace_label = " -> ".join(traces) if traces else "aucune trace"
            st.caption(f"{elapsed}s | workflow : {trace_label}")

            if show_traces and traces:
                with st.expander("Traces du workflow", expanded=False):
                    render_trace_items(traces)
            if show_traces and critic:
                with st.expander("Critique de fidelite", expanded=False):
                    st.json(critic)

        except Exception as exc:
            answer = f"Erreur : `{exc}`"
            st.error(answer)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "traces": traces,
            "critic": critic,
        }
    )
