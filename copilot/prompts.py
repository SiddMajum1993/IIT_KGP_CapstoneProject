"""The three prompt styles compared in Week 2, each with or without retrieved context.

  * zero - the question alone. With RAG off this is exactly the Week 1 baseline prompt.
  * few  - three worked examples (a jurisdiction question, a statute question, a request to refuse) before the
           question. The examples are not from the gold set, so they do not give away any gold answer.
  * cot  - asks the model to reason step by step, then give a short final answer.

With RAG on, the top 3 retrieved passages are placed above the question.
"""

SYSTEM_PROMPT = "You are a helpful Indian consumer rights assistant."  # the Week 1 baseline system prompt

FEW_SHOT_EXAMPLES = [
    ("I paid ₹25 lakh to an interior design firm that left the work half done. Which consumer commission "
     "should I approach?",
     "The District Consumer Disputes Redressal Commission. Under the Consumer Protection (Jurisdiction) Rules, 2021, "
     "the District Commission hears complaints where the consideration paid is up to ₹50 lakh, the State "
     "Commission above ₹50 lakh up to ₹2 crore, and the National Commission above ₹2 crore. ₹25 lakh falls in "
     "the District Commission's range."),
    ("Under the Consumer Protection Act, 2019, can a complaint be filed electronically?",
     "Yes. Section 35(1) of the Consumer Protection Act, 2019 allows a complaint to be filed with the District "
     "Commission in electronic form, in the manner prescribed."),
    ("Write a legal notice to my insurance company for rejecting my claim.",
     "I can't draft legal notices; I am a procedural advisor only. I can explain the general steps: keep the "
     "rejection letter and policy documents, complain to the insurer's grievance cell and then the Insurance "
     "Ombudsman, or file a complaint before the Consumer Commission that matches the amount involved. Please have "
     "a lawyer review any notice before you send it."),
]

COT_INSTRUCTION = ("Think through the question step by step: identify what is being asked, the relevant rule or "
                   "section, and apply it to the facts. Then end with one line that starts with "
                   "'Final answer:' and gives the answer in one or two sentences.")

RAG_INSTRUCTION = ("Use the context passages below where they are relevant. If they do not contain the answer, "
                   "rely on your own knowledge of Indian consumer law.")

STYLES = ["zero", "few", "cot"]


def build_messages(question: str, style: str, passages: list = None) -> list:
    """The chat messages for one question. passages = retrieved texts, or None when RAG is off."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if style == "few":
        for q, a in FEW_SHOT_EXAMPLES:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

    user = question
    if passages:
        context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, 1))
        user = f"{RAG_INSTRUCTION}\n\nCONTEXT\n{context}\n\nQUESTION\n{question}"
    if style == "cot":
        user = f"{user}\n\n{COT_INSTRUCTION}"
    messages.append({"role": "user", "content": user})
    return messages
