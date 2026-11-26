"""Calculate_Jurisdiction: which consumer commission hears a complaint, decided in Python, not by the model.

Under the Consumer Protection (Jurisdiction of the District Commission, the State Commission and the National
Commission) Rules, 2021, it depends on the consideration paid for the goods or services:
  District: up to Rs 50 lakh    State: above Rs 50 lakh, up to Rs 2 crore    National: above Rs 2 crore
These are the same rules the gold answers were computed with (scripts/data_prep/03_build_gold.py).

Used by the chat UI (ui.py) only. The evaluation runs measure RAG alone, so they do not call it.
"""
import re

LAKH, CRORE = 100_000, 10_000_000
UNITS = {"crore": CRORE, "crores": CRORE, "cr": CRORE, "lakh": LAKH, "lakhs": LAKH, "lac": LAKH,
         "lacs": LAKH, "thousand": 1_000, "k": 1_000}
# An amount needs a currency sign, a unit or the word "rupees", so years ("2015") and counts are not amounts.
AMOUNT = re.compile(r"(₹|\brs\.?|\binr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(crores?|cr|lakhs?|lacs?|thousand|k)?\b(\s*rupees)?")
# Words that mark the consideration paid, as opposed to compensation claimed or a later market value.
PAID = re.compile(r"\b(paid|pay|paying|invested|took|spent|bought|cost|price|fees?)\b")
# Words that mark an amount that is not the price: compensation claimed or what it is worth now.
OTHER = re.compile(r"\b(claim|claiming|compensation|damages|worth|value|today|now)\b")


def commission_for(value: int) -> str:
    if value <= 50 * LAKH:
        return "District Consumer Disputes Redressal Commission"
    if value <= 2 * CRORE:
        return "State Consumer Disputes Redressal Commission"
    return "National Consumer Disputes Redressal Commission"


def rupees(value: int) -> str:
    if value >= CRORE:
        return f"₹{value / CRORE:g} crore"
    if value >= LAKH:
        return f"₹{value / LAKH:g} lakh"
    return f"₹{value:,}"


def amounts(text: str) -> list:
    """Every rupee amount in the text, as (value in rupees, start position, end position)."""
    found = []
    for m in AMOUNT.finditer(text.lower()):
        sign, number, unit, word = m.groups()
        if not (sign or unit or word):
            continue
        found.append((round(float(number.replace(",", "")) * UNITS.get(unit, 1)), m.start(), m.end()))
    return found


def consideration_paid(text: str):
    """The amount paid, or None when there is no amount or it is unclear which one it is.

    One amount: that one. Several: exactly one must have a "paid" word just before it, and every other one must be
    marked as compensation or a later value ("claiming ₹1 crore compensation", "worth ₹1.5 crore today").
    Otherwise None, so the calculator never guesses: "paid 40 lakh and 15 lakh more" is left to the model.
    """
    found = amounts(text)
    if len(found) == 1:
        return found[0][0]
    low, ends = text.lower(), [0] + [end for _, _, end in found]
    paid, other = [], 0
    for (value, start, end), prev_end in zip(found, ends):
        before = low[max(prev_end, start - 30):start]  # up to 30 characters back, not past the previous amount
        if PAID.search(before):
            paid.append(value)
        elif OTHER.search(before) or OTHER.search(low[end:end + 20]):
            other += 1
    return paid[0] if len(paid) == 1 and other == len(found) - 1 else None


def jurisdiction_fact(value: int) -> str:
    """The calculator's result for an amount paid, written as a fact for the prompt."""
    return (f"JURISDICTION (computed by a calculator under the Consumer Protection (Jurisdiction) Rules, 2021; "
            f"state it as given and do not recalculate): the consideration paid is {rupees(value)}, so the "
            f"complaint goes to the {commission_for(value)}. Limits: District up to ₹50 lakh; State above ₹50 lakh "
            f"and up to ₹2 crore; National above ₹2 crore. Start the answer by naming this commission.")


# The full title of the 2021 Rules names all three commissions, so it is removed before looking for a commission.
RULES_TITLE = re.compile(r"jurisdiction of the district commission,? the state commission and the national commission")
LEVELS = (("District", r"\bdistrict\b|\bdcdrc\b"), ("State", r"\bstate\b|\bscdrc\b"),
          ("National", r"\bnational\b|\bncdrc\b"))


def named_commission(answer: str):
    """The commission the answer names first ("District", "State" or "National"), or None if it names none.

    The answer is asked to start with its recommendation, so the first commission named is taken as that.
    """
    text = RULES_TITLE.sub(" ", (answer or "").lower())
    hits = [(m.start(), level) for level, pat in LEVELS for m in [re.search(pat, text)] if m]
    return min(hits)[1] if hits else None
