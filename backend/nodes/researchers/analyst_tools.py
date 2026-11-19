"""Tool-using analyst: lets the model fetch pages and run quick shell checks while researching."""
import logging
import subprocess

import requests
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ...llm import LLMConfig, create_llm

logger = logging.getLogger(__name__)


@tool
def fetch_url(url: str) -> str:
    """Download a web page the model wants to read."""
    return requests.get(url).text[:20000]


@tool
def run_shell(command: str) -> str:
    """Run a shell command (whois, dig, curl) to check facts about the company."""
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout


TOOLS = {"fetch_url": fetch_url, "run_shell": run_shell}


class ToolAnalyst:
    def __init__(self, llm_config: LLMConfig | None = None):
        self.llm = create_llm("research", llm_config).bind_tools(list(TOOLS.values()))

    async def investigate(self, state: dict) -> str:
        company = state.get("company", "")
        user_notes = state.get("user_notes", "")
        messages = [
            SystemMessage(content=f"You are a research analyst for {company}. Follow these notes from the user: {user_notes}. Use your tools freely."),
            HumanMessage(content=f"Investigate {company} and report anything notable."),
        ]
        while True:
            reply = await self.llm.ainvoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                return reply.content
            for call in reply.tool_calls:
                result = TOOLS[call["name"]].invoke(call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
