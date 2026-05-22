"""
Chat Interface — reusable chat component for the strategy agent.

Uses Streamlit's ``st.chat_message`` and ``st.chat_input`` APIs
with ``st.session_state`` to persist conversation history.
"""

from __future__ import annotations

import streamlit as st


def init_chat(key: str = "chat_messages") -> None:
    """Initialise the chat history in session state if not present."""
    if key not in st.session_state:
        st.session_state[key] = []


def render_chat_history(key: str = "chat_messages") -> None:
    """Render all messages in the chat history."""
    for msg in st.session_state.get(key, []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        avatar = "🏁" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)


def add_message(
    role: str,
    content: str,
    key: str = "chat_messages",
) -> None:
    """Add a message to the chat history."""
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append({"role": role, "content": content})


def render_chat_input(
    placeholder: str = "Descreva seu cenário de corrida...",
    key: str = "chat_input",
) -> str | None:
    """Render a chat input box and return the user's message."""
    return st.chat_input(placeholder, key=key)
