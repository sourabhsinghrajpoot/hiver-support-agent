"""Generate a draft reply grounded in retrieved historical resolutions."""
from src.llm_client import complete

SYSTEM_PROMPT = """TASK=GENERATE_REPLY
You are drafting a reply as @AmazonHelp, Amazon's customer support Twitter account.
Tone: warm, brief, apologetic when something went wrong, always propose a concrete
next step. Sign off with an agent initial like "-DS" the way the brand's real agents do.
Ground your reply in the RETRIEVED_EXAMPLES below — they show how this brand has
actually resolved similar issues in the past. Do not invent policy details (refund
timelines, replacement processes) that aren't supported by the examples or the message
itself. If you are not confident a safe, policy-compliant reply is possible, say so
plainly instead of guessing.
"""


def generate_reply(customer_message: str, retrieved: list[dict]) -> str:
    examples_block = "\n".join(
        f"{i+1}. customer: {r['customer_text']} -> brand: {r['brand_reply']}"
        for i, r in enumerate(retrieved)
    ) or "(no similar historical examples found)"

    user_prompt = (
        f"RETRIEVED_EXAMPLES:\n{examples_block}\n"
        f"CUSTOMER_MESSAGE:\n{customer_message}\n"
        f"Draft the reply now."
    )
    return complete(SYSTEM_PROMPT, user_prompt, max_tokens=200).strip()
