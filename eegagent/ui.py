"""Streamlit front end for the eegagent API.

eegagent ui --api http://localhost:8000
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

API = os.environ.get("EEGAGENT_API", "http://localhost:8000")

st.set_page_config(page_title="EEG-Agent", page_icon="🧠", layout="wide")
st.title("EEG-Agent: source imaging by conversation")
st.caption(
    "An LLM drives MNE-Python through typed tools. Every number in the answer comes from a tool call you can inspect."
)

if "session_id" not in st.session_state:
    try:
        r = requests.post(f"{API}/sessions", timeout=30)
        r.raise_for_status()
        st.session_state.session_id = r.json()["session_id"]
        st.session_state.backend = r.json()
        st.session_state.messages = []
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API}: {e}. Start it with `eegagent serve`.")
        st.stop()

sid = st.session_state.session_id
with st.sidebar:
    st.subheader("Session")
    st.code(sid)
    st.write(
        f"provider: **{st.session_state.backend.get('provider')}**  \nmodel: `{st.session_state.backend.get('model')}`"
    )
    if st.button("New session"):
        for k in ("session_id", "messages", "backend"):
            st.session_state.pop(k, None)
        st.rerun()
    with st.expander("Settings"):
        try:
            cur = requests.get(f"{API}/settings", timeout=10).json()
        except requests.RequestException:
            cur = {"providers": ["anthropic", "ollama", "rule"], "provider": "anthropic"}
        providers = cur.get("providers", ["anthropic", "ollama", "rule"])
        provider = st.selectbox(
            "Provider", providers, index=providers.index(cur.get("provider", "anthropic"))
        )
        model = st.text_input("Model (blank = provider default)", value=cur.get("model", ""))
        key_label = "Anthropic API key" + (" (saved)" if cur.get("has_anthropic_key") else "")
        api_key = st.text_input(key_label, type="password", placeholder="leave blank to keep the saved key")
        ollama_host = st.text_input("Ollama host", value=cur.get("ollama_host", "http://localhost:11434"))
        c1, c2 = st.columns(2)
        if c1.button("Save", use_container_width=True):
            payload = {"provider": provider, "model": model, "ollama_host": ollama_host}
            if api_key:
                payload["anthropic_api_key"] = api_key
            try:
                requests.put(f"{API}/settings", json=payload, timeout=10).raise_for_status()
                st.success("Saved. Start a new session to use them.")
            except requests.RequestException as e:
                st.error(f"Could not save: {e}")
        if c2.button("Check", use_container_width=True):
            try:
                res = requests.post(f"{API}/settings/check", timeout=30).json()
                (st.success if res.get("ok") else st.error)(res.get("detail", ""))
            except requests.RequestException as e:
                st.error(f"Check failed: {e}")
        st.caption(f"Stored in {cur.get('config_path', '~/.eegagent/config.json')}")
    st.subheader("Try")
    for example in (
        "Load the simulated recording, filter 1-30 Hz, epoch -0.2 to 0.5 s and localise the event_1 response with dSPM.",
        "Test whether event_1 and event_2 differ with a cluster permutation test.",
        "Generate an HTML report of everything so far.",
    ):
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    st.subheader("State")
    try:
        st.json(requests.get(f"{API}/sessions/{sid}/state", timeout=30).json(), expanded=False)
    except requests.RequestException:
        pass

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        for ev in m.get("trace", []):
            label = f"{'❌' if ev['is_error'] else '🔧'} {ev['name']}({', '.join(f'{k}={v}' for k, v in ev['args'].items())}) · {ev['seconds']}s"
            with st.expander(label):
                st.json(ev["result"])

prompt = st.chat_input("Ask for an analysis...") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Running tools..."):
        try:
            r = requests.post(f"{API}/sessions/{sid}/chat", json={"message": prompt}, timeout=1800)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            data = {"text": f"Request failed: {e}", "trace": []}
        st.markdown(data["text"])
        for ev in data["trace"]:
            with st.expander(f"{'❌' if ev['is_error'] else '🔧'} {ev['name']} · {ev['seconds']}s"):
                st.code(json.dumps(ev["args"]), language="json")
                st.json(ev["result"])
        if any(ev["name"] == "generate_report" and not ev["is_error"] for ev in data["trace"]):
            html = requests.get(f"{API}/sessions/{sid}/report", timeout=60).text
            st.components.v1.html(html, height=700, scrolling=True)
    st.session_state.messages.append({"role": "assistant", "text": data["text"], "trace": data["trace"]})
