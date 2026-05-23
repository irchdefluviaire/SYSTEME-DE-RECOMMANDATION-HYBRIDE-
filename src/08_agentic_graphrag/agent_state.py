"""State shared by the Agentic GraphRAG LangGraph workflow."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentTrace(TypedDict, total=False):
    step: str
    status: Literal["ok", "warning", "error"]
    message: str
    details: dict[str, Any]


class AgentState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes.

    The state is deliberately explicit so LangGraph Studio can inspect each
    transition: retrieval, graph checks, scoring, critique, and generation.
    """

    user_query: str
    message: str
    message_humain: str
    question: str
    input: str
    messages: Annotated[list[AnyMessage], add_messages]
    human_message: dict[str, Any]
    candidat_id: str
    top_k: int
    mode: Literal["real"]
    intent: Literal["recommendation", "career_advice"]

    candidate_profile: dict[str, Any]
    career_guidance: dict[str, Any]
    vector_results: list[dict[str, Any]]
    graph_results: list[dict[str, Any]]
    skill_gaps: list[dict[str, Any]]
    ranked_offers: list[dict[str, Any]]
    roadmap: dict[str, Any]

    critique: dict[str, Any]
    replan_count: int
    should_replan: bool

    final_answer: str
    traces: list[AgentTrace]


class AgentInput(TypedDict, total=False):
    """External input accepted by LangGraph Studio/API.

    Keep this schema lighter than AgentState: users should be able to start a
    run with a plain text field instead of constructing the full internal state.
    """

    message: str
    message_humain: str
    question: str
    input: str
    user_query: str
    messages: Annotated[list[AnyMessage], add_messages]
    candidat_id: str
    top_k: int
    mode: Literal["real"]
