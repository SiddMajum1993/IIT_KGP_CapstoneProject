"""Copilot chat UI: ask a question, get the answer from the full pipeline (retrieval -> prompt -> model).

Run:  streamlit run ui.py

- Phone numbers and e-mail addresses are removed before the question reaches the model (app.scrub).
- The retriever is built once per server (st.cache_resource); search results are cached per question (st.cache_data).
- A topic check runs first: an off-topic question gets a fixed reply, with no search and no passages.
- Jurisdiction calculator (copilot/jurisdiction.py): if the question names the amount paid, Python decides the
  commission under the 2021 Rules and the model is given that result as a fact, so it does not do the arithmetic.
- The answer is streamed, so it appears as the model writes it, and is asked to stay within about 150 words.
- At most 5 questions per session, as the PRD requires.
"""
import time

import streamlit as st

from app import DISCLAIMER, scrub
from copilot import config, jurisdiction, llm, prompts, rag

MAX_QUESTIONS = 5
PROMPT_NAMES = {"zero": "Zero-shot", "few": "Few-shot", "cot": "Chain-of-thought"}
# UI only: the evaluation runs (copilot/run.py) skip the topic check and keep the unchanged prompts,
# so their results stay comparable.
UI_RULES = "Keep the answer short: at most about 150 words."
TOPIC_CHECK = ("You decide whether a message is about consumer rights in India: a complaint about a product or "
               "service, the Consumer Protection Act, consumer commissions, or a request for legal help in a "
               "consumer dispute. Reply with one word: yes or no.")
OFF_TOPIC_REPLY = "Sorry, I can only help with questions about Indian consumer rights."


@st.cache_resource(show_spinner=False)
def get_retriever(mode: str) -> rag.Retriever:
    return rag.load_retriever(mode)


@st.cache_data(show_spinner=False)
def retrieve(mode: str, question: str) -> list:
    """The top 3 passages as (master-set row id, text) pairs."""
    return [(d.metadata["qa_id"], d.page_content) for d in get_retriever(mode).search(question)]


def is_on_topic(mode: str, question: str) -> bool:
    """One short model call before anything else. If the check itself fails, the question is let through."""
    client, model, extra_body = llm.make_client(mode)
    reply = llm.chat(client, model, extra_body, [{"role": "system", "content": TOPIC_CHECK},
                                                 {"role": "user", "content": question}])["answer"] or ""
    return not reply.strip().lower().startswith("no")


def stream_answer(mode: str, style: str, question: str, passages: list, fact: str = None) -> dict:
    """Write the model's answer to the page as it arrives; return it for the chat history."""
    client, model, extra_body = llm.make_client(mode)
    messages = prompts.build_messages(question, style, passages or None)
    messages[0] = dict(role="system", content=f"{messages[0]['content']} {UI_RULES}")
    if fact:  # the calculator's result goes first in the question, above any passages
        messages[-1] = dict(role="user", content=f"{fact}\n\n{messages[-1]['content']}")
    start = time.perf_counter()
    try:
        text = st.write_stream(llm.stream(client, model, extra_body, messages))
    except Exception as e:  # noqa: BLE001 - show any model/API failure in the chat instead of crashing
        text = f"Error from the model: {e}"
        st.error(text)
    return dict(model=model, text=text, latency=round(time.perf_counter() - start, 1))


def show_sources(sources: list) -> None:
    with st.expander(f"Sources ({len(sources)} passages)"):
        for i, (qa_id, text) in enumerate(sources, 1):
            st.markdown(f"**[{i}] {qa_id}**")
            st.write(text)


def show_turn(turn: dict) -> None:
    """Redraw a finished question and answer from the chat history."""
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("off_topic"):
            st.markdown(OFF_TOPIC_REPLY)
            st.caption("off-topic, not searched")
            return
        if turn.get("fact"):
            st.info(f"🧮 {turn['fact_short']}")
        shown = turn["rag"] or turn["plain"]
        st.markdown(shown["text"])
        st.caption(f"{shown['model']} · {shown['latency']}s · {'with' if turn['rag'] else 'without'} RAG")
        if turn["rag"]:
            show_sources(turn["sources"])
        if turn["rag"] and turn["plain"]:
            with st.expander("Answer without RAG (for comparison)"):
                st.markdown(turn["plain"]["text"])


def main() -> None:
    st.set_page_config(page_title="Consumer Rights Copilot", page_icon="⚖️")
    st.title("⚖️ Indian Consumer Rights Copilot")
    st.warning(DISCLAIMER)

    with st.sidebar:
        mode = st.radio("Mode", ["local", "api"], index=["local", "api"].index(config.get("COPILOT_MODE")))
        style = st.radio("Prompt style", prompts.STYLES, format_func=PROMPT_NAMES.get)
        use_rag = st.toggle("Use RAG", value=True)
        compare = st.toggle("Also show the answer without RAG", value=False, disabled=not use_rag)
        use_calc = st.toggle("Jurisdiction calculator", value=True,
                             help="Decide the commission in Python from the amount paid (2021 Rules).")
        st.session_state.setdefault("turns", [])
        used = len(st.session_state.turns)
        st.metric("Questions this session", f"{used} / {MAX_QUESTIONS}")

    if use_rag:
        with st.spinner("Loading the retriever (first start only, about a minute)..."):
            get_retriever(mode)

    for turn in st.session_state.turns:
        show_turn(turn)

    if used >= MAX_QUESTIONS:
        st.info(f"You have used all {MAX_QUESTIONS} questions for this session.")
        return
    question = st.chat_input("Ask about the Consumer Protection Act, 2019...")
    if not question:
        return

    question = scrub(question)
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the question..."):
            on_topic = is_on_topic(mode, question)
        if not on_topic:
            st.session_state.turns.append(dict(question=question, off_topic=True))
            st.rerun()
        value = jurisdiction.consideration_paid(question) if use_calc else None
        fact = fact_short = None
        if value is not None:
            fact = jurisdiction.jurisdiction_fact(value)
            fact_short = (f"Jurisdiction calculator: {jurisdiction.rupees(value)} paid → "
                          f"{jurisdiction.commission_for(value)}")
            st.info(f"🧮 {fact_short}")
        sources = []
        if use_rag:
            with st.status("Searching the legal corpus...", expanded=False) as status:
                start = time.perf_counter()
                sources = retrieve(mode, question)
                status.update(label=f"Found {len(sources)} passages in {time.perf_counter() - start:.1f}s",
                              state="complete")
        rag_answer = stream_answer(mode, style, question, [t for _, t in sources], fact) if use_rag else None
        if rag_answer:
            show_sources(sources)
        plain = None
        if not use_rag or compare:
            if use_rag:
                st.markdown("**Answer without RAG (for comparison)**")
            plain = stream_answer(mode, style, question, [], fact)
    st.session_state.turns.append(dict(question=question, sources=sources, rag=rag_answer, plain=plain,
                                       fact=fact, fact_short=fact_short))
    st.rerun()


main()
