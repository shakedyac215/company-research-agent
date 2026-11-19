"""SoreqAI PR-flow test helper (scenario 8): used by the triage workflow. Do not merge."""


def triage_label(body: str) -> str:
    return "bug" if "error" in body.lower() else "question"
