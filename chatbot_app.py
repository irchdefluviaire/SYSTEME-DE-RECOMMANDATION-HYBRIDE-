"""
Interface Streamlit moderne — Conseiller Emploi-Compétences KmerAI.

Lancement :
    poetry run streamlit run chatbot_app.py --server.port=8501 --server.address=127.0.0.1
"""

from __future__ import annotations

import base64
import html
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
    _module_path_str = str(_module_path)
    if _module_path_str not in sys.path:
        sys.path.insert(0, _module_path_str)


st.set_page_config(
    page_title="Conseiller Emploi-Compétences — KmerAI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "KmerAI — assistant de recherche et d'orientation emploi-compétences. "
            "Les résultats restent soumis au jugement humain."
        )
    },
)


APP_CSS = """
<style>
:root {
    --app-bg: var(--background-color, #f5f7fb);
    --app-surface: var(--secondary-background-color, #eef2f7);
    --app-text: var(--text-color, #07152f);
    --app-muted: color-mix(in srgb, var(--app-text) 62%, var(--app-bg));
    --app-subtle: color-mix(in srgb, var(--app-text) 46%, var(--app-bg));
    --surface-raised: color-mix(in srgb, var(--app-bg) 94%, var(--app-text));
    --surface-soft: color-mix(in srgb, var(--app-surface) 82%, var(--app-bg));
    --surface-glass: color-mix(in srgb, var(--app-bg) 91%, transparent);
    --line: color-mix(in srgb, var(--app-text) 15%, transparent);
    --line-strong: color-mix(in srgb, var(--app-text) 24%, transparent);
    --teal: #008a91;
    --teal-bright: #2ac8c3;
    --teal-soft: color-mix(in srgb, var(--teal) 14%, var(--app-bg));
    --magenta: #a40e61;
    --green-soft: color-mix(in srgb, #23a978 15%, var(--app-bg));
    --amber-soft: color-mix(in srgb, #e2a100 17%, var(--app-bg));
    --shadow-sm: 0 8px 24px rgba(3, 12, 28, 0.10);
    --shadow-md: 0 18px 46px rgba(3, 12, 28, 0.18);
}

html, body, [class*="css"] {
    font-family: "Segoe UI", Inter, Arial, sans-serif;
}

.stApp {
    color: var(--app-text);
    background:
        radial-gradient(circle at 78% -12%, rgba(0, 138, 145, 0.11), transparent 26rem),
        radial-gradient(circle at 8% 12%, rgba(164, 14, 97, 0.055), transparent 22rem),
        var(--app-bg);
}

[data-testid="stHeader"] { background: transparent; }

[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding-top: 1.15rem;
    padding-bottom: 7rem;
}

[data-testid="stSidebar"] {
    min-width: 316px;
    max-width: 316px;
    background: var(--surface-glass);
    border-right: 1px solid var(--line);
    backdrop-filter: blur(18px);
}

[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.35rem 0 0.85rem;
}

.brand-lockup img {
    width: 52px;
    height: 52px;
    object-fit: contain;
    border-radius: 14px;
    background: var(--surface-raised);
}

.brand-lockup strong {
    display: block;
    color: var(--app-text);
    font-size: 1.02rem;
    letter-spacing: -0.02em;
}

.brand-lockup span {
    display: block;
    color: var(--app-muted);
    font-size: 0.76rem;
    margin-top: 0.12rem;
}

.workspace-label {
    color: var(--app-muted);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    margin: 0.35rem 0 0.5rem;
    text-transform: uppercase;
}

.candidate-context {
    background: linear-gradient(145deg, var(--teal-soft), var(--surface-raised));
    border: 1px solid color-mix(in srgb, var(--teal) 28%, transparent);
    border-radius: 14px;
    margin: 0.45rem 0 0.8rem;
    padding: 0.78rem 0.9rem;
}

.candidate-context small {
    color: var(--app-muted);
    display: block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.candidate-context strong {
    color: var(--app-text);
    display: block;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.76rem;
    margin-top: 0.28rem;
    overflow-wrap: anywhere;
}

.privacy-note {
    color: var(--app-muted);
    font-size: 0.72rem;
    line-height: 1.45;
    margin-top: 0.85rem;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    min-height: 220px;
    padding: 2rem 2.1rem;
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 26px;
    background:
        radial-gradient(circle at 88% 20%, rgba(24, 194, 183, 0.34), transparent 19rem),
        linear-gradient(128deg, #071b41 0%, #0b3267 53%, #006f79 100%);
    box-shadow: var(--shadow-md);
    color: #fff;
    isolation: isolate;
}

.hero-shell::after {
    content: "";
    position: absolute;
    right: -5rem;
    bottom: -8rem;
    width: 21rem;
    height: 21rem;
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 50%;
    box-shadow: 0 0 0 3rem rgba(255, 255, 255, 0.035),
                0 0 0 6rem rgba(255, 255, 255, 0.02);
    z-index: -1;
}

.hero-topline {
    align-items: center;
    display: flex;
    gap: 0.55rem;
    margin-bottom: 0.85rem;
}

.hero-eyebrow {
    color: #aef2eb;
    font-size: 0.71rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #53e6bd;
    box-shadow: 0 0 0 5px rgba(83, 230, 189, 0.14);
}

.hero-shell h1 {
    max-width: 760px;
    color: #fff !important;
    font-size: clamp(2rem, 4.2vw, 3.35rem) !important;
    font-weight: 780 !important;
    letter-spacing: -0.048em !important;
    line-height: 1.02 !important;
    margin: 0 !important;
}

.hero-shell p {
    max-width: 700px;
    color: #d9e9f7 !important;
    font-size: 0.98rem;
    line-height: 1.55;
    margin: 0.85rem 0 1.2rem !important;
}

.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.48rem;
}

.hero-chip {
    padding: 0.34rem 0.65rem;
    color: #eafcff;
    border: 1px solid rgba(220, 252, 255, 0.22);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(8px);
    font-size: 0.72rem;
    font-weight: 650;
}

.section-kicker {
    color: color-mix(in srgb, var(--magenta) 82%, var(--app-text));
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    margin: 1.45rem 0 0.22rem;
    text-transform: uppercase;
}

.section-title {
    color: var(--app-text);
    font-size: 1.34rem;
    font-weight: 760;
    letter-spacing: -0.025em;
    margin: 0 0 0.2rem;
}

.section-copy {
    color: var(--app-muted);
    font-size: 0.86rem;
    margin-bottom: 0.82rem;
}

.trust-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 1rem 0 0.55rem;
}

.trust-card {
    min-height: 118px;
    padding: 1rem;
    border: 1px solid var(--line);
    border-radius: 17px;
    background: var(--surface-raised);
    box-shadow: var(--shadow-sm);
}

.trust-icon {
    align-items: center;
    display: flex;
    justify-content: center;
    width: 34px;
    height: 34px;
    color: var(--teal);
    border-radius: 10px;
    background: var(--teal-soft);
    font-size: 1rem;
}

.trust-card strong {
    display: block;
    color: var(--app-text);
    font-size: 0.86rem;
    margin: 0.7rem 0 0.25rem;
}

.trust-card span {
    color: var(--app-muted);
    font-size: 0.74rem;
    line-height: 1.45;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    min-height: 2.55rem;
    color: var(--app-text);
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--surface-raised);
    box-shadow: 0 3px 12px rgba(10, 35, 78, 0.045);
    font-weight: 650;
    transition: border-color 150ms ease, box-shadow 150ms ease, transform 150ms ease;
}

div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    color: var(--app-text);
    border-color: color-mix(in srgb, var(--teal) 58%, var(--line));
    box-shadow: 0 8px 20px rgba(10, 35, 78, 0.09);
    transform: translateY(-1px);
}

div[data-testid="stButton"] > button:focus-visible,
div[data-testid="stDownloadButton"] > button:focus-visible {
    outline: 3px solid rgba(0, 138, 145, 0.25);
    outline-offset: 2px;
}

div[data-testid="stButton"] > button[kind="primary"] {
    color: #fff;
    border-color: #0b2555;
    background: linear-gradient(130deg, #0b2555, #0b4776);
}

div[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div {
    border-color: var(--line);
    border-radius: 11px;
    background: var(--surface-raised);
    color: var(--app-text);
}

div[data-testid="stChatMessage"] {
    width: fit-content;
    max-width: min(92%, 920px);
    padding: 1rem 1.12rem;
    border: 1px solid var(--line);
    border-radius: 20px 20px 20px 7px;
    background: var(--surface-raised);
    box-shadow: var(--shadow-sm);
    margin: 0 0 0.9rem 0;
    color: var(--app-text);
}

div[data-testid="stChatMessage"]:has(.message-role-user) {
    max-width: min(78%, 760px);
    margin-left: auto;
    border-color: color-mix(in srgb, var(--teal) 35%, var(--line));
    border-radius: 20px 20px 7px 20px;
    background: linear-gradient(145deg, var(--teal-soft), var(--surface-raised));
}

div[data-testid="stChatMessage"]:has(.message-role-assistant) {
    margin-right: auto;
}

div[data-testid="stChatMessage"] [data-testid*="Avatar"] {
    width: 2.25rem;
    height: 2.25rem;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--app-surface);
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] code {
    color: var(--app-text);
}

.message-role {
    display: flex;
    align-items: center;
    gap: 0.42rem;
    margin: 0 0 0.52rem;
    color: var(--app-muted);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.075em;
    line-height: 1;
    text-transform: uppercase;
}

.message-role-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--teal-bright);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--teal) 12%, transparent);
}

.message-role-user {
    justify-content: flex-end;
    color: color-mix(in srgb, var(--teal) 76%, var(--app-text));
}

.message-role-user .message-role-dot {
    order: 2;
    background: var(--magenta);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--magenta) 12%, transparent);
}

div[data-testid="stChatInput"] {
    border-top: 0;
    background: color-mix(in srgb, var(--app-bg) 88%, transparent);
    backdrop-filter: blur(14px);
}

div[data-testid="stChatInput"] > div {
    max-width: 1120px;
    margin: 0 auto 0.75rem;
    border: 1px solid var(--line-strong);
    border-radius: 17px;
    background: var(--surface-raised);
    box-shadow: 0 14px 36px rgba(3, 12, 28, 0.20);
}

div[data-testid="stChatInput"] textarea {
    color: var(--app-text) !important;
    caret-color: var(--teal-bright);
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: var(--app-subtle) !important;
}

div[data-testid="stStatusWidget"] {
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--teal) 28%, transparent) !important;
    border-radius: 14px !important;
    background: linear-gradient(145deg, var(--teal-soft), var(--surface-raised)) !important;
}

.message-meta {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.38rem;
    padding-top: 0.62rem;
}

.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    padding: 0.27rem 0.52rem;
    color: var(--app-muted);
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--surface-soft);
    font-size: 0.67rem;
    font-weight: 700;
    white-space: nowrap;
}

.meta-chip.success {
    color: color-mix(in srgb, #18a774 78%, var(--app-text));
    border-color: color-mix(in srgb, #18a774 38%, transparent);
    background: var(--green-soft);
}

.meta-chip.warning {
    color: color-mix(in srgb, #d89500 76%, var(--app-text));
    border-color: color-mix(in srgb, #d89500 42%, transparent);
    background: var(--amber-soft);
}

.step-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.48rem;
    color: var(--app-muted);
    border-radius: 999px;
    background: var(--app-surface);
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.64rem;
    font-weight: 650;
    white-space: nowrap;
}

.disclaimer {
    display: flex;
    gap: 0.55rem;
    margin: 0.85rem 0 0.35rem;
    padding: 0.72rem 0.85rem;
    color: var(--app-muted);
    border: 1px solid var(--line);
    border-radius: 13px;
    background: var(--surface-glass);
    font-size: 0.72rem;
    line-height: 1.45;
}

@media (max-width: 800px) {
    [data-testid="stMainBlockContainer"] { padding-top: 0.55rem; }
    .hero-shell { min-height: 0; padding: 1.4rem; border-radius: 20px; }
    .hero-shell h1 { font-size: 2rem !important; }
    .trust-grid { grid-template-columns: 1fr; }
    div[data-testid="stChatMessage"],
    div[data-testid="stChatMessage"]:has(.message-role-user) {
        width: 100%;
        max-width: 100%;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


LOGO_KMERAI = FIGURES / "kmerai.png"

STEP_LABELS: dict[str, str] = {
    "analyse_request": "Compréhension de la demande",
    "plan_tools": "Choix des sources et outils",
    "execute_tools": "Recherche des informations",
    "build_context": "Assemblage des preuves",
    "generate_final_answer": "Rédaction de la réponse",
    "expand_context": "Élargissement de la recherche",
    "critic": "Contrôle de l'ancrage",
}

SUGGESTIONS: tuple[dict[str, str], ...] = (
    {
        "icon": "◎",
        "label": "Recommander des offres",
        "prompt": "Montre-moi les meilleures offres pour PPKOU2501080016340",
    },
    {
        "icon": "↗",
        "label": "Analyser un skill gap",
        "prompt": (
            "Analyse le skill gap du candidat PPKOU2501080016340 "
            "et propose une roadmap de progression"
        ),
    },
    {
        "icon": "◇",
        "label": "Explorer un métier",
        "prompt": "Je veux devenir data analyst dans une banque camerounaise",
    },
    {
        "icon": "▦",
        "label": "Consulter un référentiel",
        "prompt": "Explique la classification NCF pour le domaine informatique",
    },
)


@st.cache_resource(show_spinner=False)
def load_graph():
    """Charge une seule fois le workflow LangGraph et ses dépendances."""
    from graph import graph as langgraph_app  # noqa: PLC0415

    return langgraph_app


@st.cache_data(show_spinner=False)
def image_data_uri(path: str) -> str:
    image_path = Path(path)
    if not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_input", "")
    st.session_state.setdefault("feedback", {})


def reset_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["pending_input"] = ""
    st.session_state["feedback"] = {}


def conversation_as_markdown(messages: list[dict[str, Any]]) -> str:
    lines = [
        "# Conversation — Conseiller Emploi-Compétences KmerAI",
        "",
        f"Exportée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
        "",
        "> Outil d'aide à la décision : les recommandations nécessitent une vérification humaine.",
        "",
    ]
    for message in messages:
        role = "Utilisateur" if message.get("role") == "user" else "Conseiller KmerAI"
        lines.extend([f"## {role}", "", str(message.get("content", "")), ""])
    return "\n".join(lines)


def render_trace_items(traces: list[str]) -> None:
    for trace in traces:
        st.code(str(trace), language="text")


def timing_html(timing_steps: list[dict[str, Any]], elapsed: float) -> str:
    chips = [f'<span class="meta-chip">⏱ {elapsed:.1f} s</span>']
    chips.extend(
        '<span class="step-badge">'
        f'{html.escape(str(step.get("short_label") or step.get("label", "Étape")))} '
        f'· {float(step.get("dt", 0.0)):.2f} s</span>'
        for step in timing_steps
    )
    return '<div class="message-meta">' + "".join(chips) + "</div>"


def critic_html(critic: dict[str, Any]) -> str:
    if not critic:
        return ""

    decision = str(critic.get("decision", "")).lower()
    faithfulness = critic.get("faithfulness")
    source = critic.get("source")
    chips: list[str] = []

    if decision == "accept":
        chips.append(
            '<span class="meta-chip success" title="Réponse suffisamment ancrée '
            'dans le contexte récupéré ; cela ne constitue pas une validation métier.">'
            "✓ Contexte vérifié</span>"
        )
    elif decision == "revise":
        chips.append(
            '<span class="meta-chip warning" title="Le système a demandé une révision '
            'ou davantage de contexte.">↻ Réponse révisée</span>'
        )

    if isinstance(faithfulness, (int, float)):
        score = max(0.0, min(float(faithfulness), 1.0))
        chips.append(
            '<span class="meta-chip" title="Mesure expérimentale de recouvrement '
            'avec le contexte, et non score de vérité métier.">'
            f"Ancrage lexical · {score:.0%}</span>"
        )

    if source:
        chips.append(
            f'<span class="meta-chip">Source moteur · {html.escape(str(source))}</span>'
        )

    if not chips:
        return ""
    return '<div class="message-meta">' + "".join(chips) + "</div>"


def queue_prompt(prompt: str) -> None:
    st.session_state["pending_input"] = prompt


def render_brand_lockup() -> None:
    logo_uri = image_data_uri(str(LOGO_KMERAI))
    logo = f'<img src="{logo_uri}" alt="Logo KmerAI">' if logo_uri else ""
    st.markdown(
        f"""
        <div class="brand-lockup">
            {logo}
            <div>
                <strong>KmerAI Career</strong>
                <span>Conseiller emploi-compétences</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, int, bool]:
    with st.sidebar:
        render_brand_lockup()
        st.markdown('<div class="workspace-label">Contexte de travail</div>', unsafe_allow_html=True)

        candidat_id = st.text_input(
            "Identifiant candidat",
            placeholder="Ex. PPKOU2501080016340",
            help=(
                "Facultatif pour l'orientation générale. Renseignez-le pour obtenir "
                "un matching et un skill gap personnalisés."
            ),
        ).strip()

        if candidat_id:
            st.markdown(
                '<div class="candidate-context"><small>Profil actif</small>'
                f'<strong>{html.escape(candidat_id)}</strong></div>',
                unsafe_allow_html=True,
            )

        top_k = st.select_slider(
            "Nombre de résultats",
            options=[3, 5, 10, 15, 20],
            value=5,
            help="Nombre maximal d'offres ou d'entités restituées.",
        )

        with st.expander("Réglages avancés", expanded=False):
            show_traces = st.toggle(
                "Afficher les détails techniques",
                value=False,
                help="Affiche les traces LangGraph et le diagnostic brut du critic.",
            )
            st.caption(
                "Le critic mesure l'ancrage lexical de la réponse dans les preuves "
                "récupérées. Il ne juge pas la vérité métier."
            )

        st.markdown('<div class="workspace-label">Conversation</div>', unsafe_allow_html=True)
        if st.button(
            "＋ Nouvelle conversation",
            type="primary",
            use_container_width=True,
        ):
            reset_conversation()
            st.rerun()

        messages = st.session_state.get("messages", [])
        st.download_button(
            "↓ Exporter en Markdown",
            data=conversation_as_markdown(messages),
            file_name=f"conversation_kmerai_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not bool(messages),
        )

        st.markdown(
            """
            <div class="privacy-note">
                <strong>Confidentialité</strong><br>
                Évitez de saisir des informations personnelles sensibles. Les résultats
                servent à l'orientation et ne remplacent ni un recruteur ni un conseiller.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return candidat_id, int(top_k), bool(show_traces)


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-topline">
                <span class="live-dot"></span>
                <span class="hero-eyebrow">Agentic GraphRAG · Cameroun</span>
            </div>
            <h1>Votre trajectoire professionnelle, éclairée par les compétences.</h1>
            <p>
                Explorez des offres, mesurez vos écarts de compétences et construisez
                un parcours de progression à partir de données structurées et de preuves
                traçables.
            </p>
            <div class="hero-chips">
                <span class="hero-chip">Recherche sémantique</span>
                <span class="hero-chip">Graphe de connaissances</span>
                <span class="hero-chip">Skill gap explicable</span>
                <span class="hero-chip">Décision humaine</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="section-kicker">Commencer une analyse</div>
        <div class="section-title">Que souhaitez-vous explorer ?</div>
        <div class="section-copy">
            Choisissez une action ou posez directement votre question dans la zone de saisie.
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        with columns[index % 2]:
            if st.button(
                f'{suggestion["icon"]}  {suggestion["label"]}',
                help=suggestion["prompt"],
                use_container_width=True,
                key=f"suggestion_{index}",
            ):
                queue_prompt(suggestion["prompt"])
                st.rerun()

    st.markdown(
        """
        <div class="trust-grid">
            <div class="trust-card">
                <div class="trust-icon">◎</div>
                <strong>Matching hybride</strong>
                <span>La proximité sémantique est complétée par les relations entre métiers,
                compétences, offres et formations.</span>
            </div>
            <div class="trust-card">
                <div class="trust-icon">⌁</div>
                <strong>Réponse traçable</strong>
                <span>Le workflow assemble un contexte avant de générer sa réponse et peut
                demander une révision.</span>
            </div>
            <div class="trust-card">
                <div class="trust-icon">◇</div>
                <strong>Humain aux commandes</strong>
                <span>Les scores soutiennent une décision ; ils ne recrutent, n'orientent et
                n'excluent personne automatiquement.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(
    message: dict[str, Any],
    index: int,
    *,
    show_traces: bool,
) -> None:
    role = str(message.get("role", "assistant"))
    is_user = role == "user"
    avatar = "🧑🏾‍💼" if is_user else ":material/auto_awesome:"
    with st.chat_message(role, avatar=avatar):
        role_class = "user" if is_user else "assistant"
        role_label = "Vous" if is_user else "Conseiller KmerAI"
        st.markdown(
            '<div class="message-role message-role-'
            f'{role_class}"><span class="message-role-dot"></span>'
            f"<span>{role_label}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(str(message.get("content", "")))

        if role == "assistant":
            timing = message.get("timing") or []
            elapsed = message.get("elapsed")
            if timing and isinstance(elapsed, (int, float)):
                st.markdown(
                    timing_html(timing, float(elapsed)),
                    unsafe_allow_html=True,
                )

            critic = message.get("critic") or {}
            critic_markup = critic_html(critic)
            if critic_markup:
                st.markdown(critic_markup, unsafe_allow_html=True)

            feedback_key = f"message_{message.get('id', index)}"
            current_feedback = st.session_state["feedback"].get(feedback_key)
            action_left, action_mid, _ = st.columns([1.1, 1.1, 7.8])
            with action_left:
                if st.button(
                    "Utile" if current_feedback != "up" else "✓ Utile",
                    key=f"up_{feedback_key}",
                    use_container_width=True,
                ):
                    st.session_state["feedback"][feedback_key] = "up"
                    st.toast("Merci pour votre retour.", icon="✅")
            with action_mid:
                if st.button(
                    "À revoir" if current_feedback != "down" else "✓ À revoir",
                    key=f"down_{feedback_key}",
                    use_container_width=True,
                ):
                    st.session_state["feedback"][feedback_key] = "down"
                    st.toast("Retour enregistré pour cette session.", icon="🔄")

        if show_traces and message.get("traces"):
            with st.expander("Journal technique du workflow", expanded=False):
                render_trace_items([str(item) for item in message["traces"]])
        if show_traces and message.get("critic"):
            with st.expander("Diagnostic brut du critic", expanded=False):
                st.json(message["critic"])


def extract_final_answer(last_state: dict[str, Any] | None) -> str:
    if not last_state:
        return ""
    for message in reversed(last_state.get("messages", [])):
        if (
            hasattr(message, "content")
            and message.content
            and getattr(message, "type", "") == "ai"
        ):
            return str(message.content)
    return ""


initialize_state()
candidat_id_input, top_k, show_traces = render_sidebar()
render_hero()

messages: list[dict[str, Any]] = st.session_state["messages"]
if not messages:
    render_empty_state()
else:
    st.markdown(
        """
        <div class="section-kicker">Conversation active</div>
        <div class="section-title">Analyse et recommandations</div>
        """,
        unsafe_allow_html=True,
    )

for message_index, stored_message in enumerate(messages):
    render_message(stored_message, message_index, show_traces=show_traces)

st.markdown(
    """
    <div class="disclaimer">
        <span>ⓘ</span>
        <span>Les recommandations sont des aides à l'exploration fondées sur les données
        disponibles. Vérifiez les exigences de chaque offre et gardez une décision humaine.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

pending_input = str(st.session_state.pop("pending_input", "") or "")
input_placeholder = (
    f"Interroger le profil {candidat_id_input}…"
    if candidat_id_input
    else "Posez une question sur un métier, une offre ou vos compétences…"
)
typed_input = st.chat_input(input_placeholder)
user_input = (typed_input or pending_input).strip()

if not user_input:
    st.stop()

message_id = int(time.time_ns())
user_message = {"id": f"u_{message_id}", "role": "user", "content": user_input}
with st.chat_message("user", avatar="🧑🏾‍💼"):
    st.markdown(
        '<div class="message-role message-role-user">'
        '<span class="message-role-dot"></span><span>Vous</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(user_input)
st.session_state["messages"].append(user_message)

answer = ""
traces: list[str] = []
critic: dict[str, Any] = {}
timing_steps: list[dict[str, Any]] = []
elapsed = 0.0

with st.chat_message("assistant", avatar=":material/auto_awesome:"):
    st.markdown(
        '<div class="message-role message-role-assistant">'
        '<span class="message-role-dot"></span><span>Conseiller KmerAI</span></div>',
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Initialisation du moteur de recherche hybride…"):
            graph = load_graph()

        agent_input: dict[str, Any] = {
            "messages": [("user", user_input)],
            "top_k": top_k,
        }
        if candidat_id_input:
            agent_input["candidat_id"] = candidat_id_input

        start_time = time.perf_counter()
        previous_time = start_time
        last_state: dict[str, Any] | None = None

        with st.status("Analyse de votre demande…", expanded=True) as status_box:
            for step in graph.stream(agent_input):
                node_name = next(iter(step))
                state = step[node_name]

                if node_name == "__end__":
                    last_state = state
                    continue

                now = time.perf_counter()
                step_elapsed = now - previous_time
                previous_time = now
                label = STEP_LABELS.get(node_name, node_name.replace("_", " ").title())

                timing_steps.append(
                    {
                        "label": label,
                        "short_label": label.split(" ")[0],
                        "dt": step_elapsed,
                    }
                )

                if show_traces:
                    state_traces = state.get("traces", [])
                    detail = str(state_traces[-1]) if state_traces else ""
                    st.write(f"**{label}** · {step_elapsed:.2f} s  ")
                    if detail:
                        st.caption(detail)
                else:
                    st.write(f"**{label}** · {step_elapsed:.2f} s")

                last_state = state

            elapsed = round(time.perf_counter() - start_time, 1)
            status_box.update(
                label=f"Analyse terminée en {elapsed:.1f} s",
                state="complete",
                expanded=False,
            )

        answer = extract_final_answer(last_state)
        if last_state:
            traces = [str(item) for item in last_state.get("traces", [])]
            critic = dict(last_state.get("critic", {}) or {})

        if not answer:
            answer = (
                "Je n'ai pas pu produire une réponse exploitable avec le contexte "
                "disponible. Reformulez la demande ou précisez le métier, l'offre ou "
                "l'identifiant candidat concerné."
            )

        st.markdown(answer)
        if timing_steps:
            st.markdown(timing_html(timing_steps, elapsed), unsafe_allow_html=True)
        critic_markup = critic_html(critic)
        if critic_markup:
            st.markdown(critic_markup, unsafe_allow_html=True)

        if show_traces and traces:
            with st.expander("Journal technique du workflow", expanded=False):
                render_trace_items(traces)
        if show_traces and critic:
            with st.expander("Diagnostic brut du critic", expanded=False):
                st.json(critic)

    except Exception as exc:  # l'interface doit rester utilisable si un backend tombe
        answer = (
            "Le moteur n'est pas disponible pour le moment. Vérifiez la connexion aux "
            "services de données, puis réessayez."
        )
        st.error(answer, icon="⚠️")
        if show_traces:
            with st.expander("Détail technique", expanded=False):
                st.exception(exc)

assistant_message = {
    "id": f"a_{message_id}",
    "role": "assistant",
    "content": answer,
    "traces": traces,
    "critic": critic,
    "timing": timing_steps,
    "elapsed": elapsed,
}
st.session_state["messages"].append(assistant_message)
