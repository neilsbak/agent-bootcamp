"""Reason-and-Act AML Knowledge Retrieval Agent (CLI + Library)."""

import asyncio
import contextlib
import csv
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import agents
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.prompts import REACT_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
)

import argparse


# ---------------------------------------------------------------------------
# Setup & globals
# ---------------------------------------------------------------------------

load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO)

AGENT_LLM_NAME = "gemini-2.5-flash"

configs = Configs.from_env_var()

async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)

async_openai_client = AsyncOpenAI()

async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="canada_aml_policies",
)


@dataclass
class AMLInputRow:
    client_id: str
    client_name: str
    pep_flag: int
    sanctions_flag: int
    fatf_country_flag: int
    ofac_country_flag: int
    transaction_amount: float


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

async def _cleanup_clients() -> None:
    """Close async clients."""
    with contextlib.suppress(Exception):
        await async_weaviate_client.close()
    with contextlib.suppress(Exception):
        await async_openai_client.close()


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _row_from_dict(row: Dict[str, Any]) -> AMLInputRow:
    """
    Convert a DictReader row into AMLInputRow.

    Assumes CSV headers:
        client_id, client_name,
        pep_flag, sanctions_flag, fatf_country_flag, ofac_country_flag,
        transaction_amount (or 'transaction amount')
    """
    # be a bit forgiving on the transaction column name
    tx_key = "transaction_amount"
    if tx_key not in row:
        # fall back to "transaction amount" if that's what the CSV uses
        if "transaction amount" in row:
            tx_key = "transaction amount"
        else:
            raise KeyError(
                "CSV must contain 'transaction_amount' or 'transaction amount' column."
            )

    return AMLInputRow(
        client_id=str(row["client_id"]).strip(),
        client_name=str(row["client_name"]).strip(),
        pep_flag=int(row["pep_flag"]),
        sanctions_flag=int(row["sanctions_flag"]),
        fatf_country_flag=int(row["fatf_country_flag"]),
        ofac_country_flag=int(row["ofac_country_flag"]),
        transaction_amount=float(row[tx_key]),
    )


def build_aml_query(input_row: AMLInputRow) -> str:
    """
    Build the natural language query for the agent from a single AMLInputRow.

    Uses the *_flag columns (0/1) and transaction_amount to describe the risk profile.
    """
    flag_labels = []
    if input_row.pep_flag == 1:
        flag_labels.append("PEP exposure")
    if input_row.sanctions_flag == 1:
        flag_labels.append("sanctions list")
    if input_row.fatf_country_flag == 1:
        flag_labels.append("high-risk / monitored FATF country")
    if input_row.ofac_country_flag == 1:
        flag_labels.append("OFAC-related country risk")

    if flag_labels:
        flags_sentence = (
            f"The client has the following AML risk flags set to 1: {', '.join(flag_labels)}."
        )
    else:
        flags_sentence = (
            "The client has no PEP, sanctions, FATF-country, or OFAC-country flags "
            "(all corresponding flag fields are 0)."
        )

    query = (
        f"Client {input_row.client_name} (client_id: {input_row.client_id}) "
        f"is making a transaction of {input_row.transaction_amount:.2f} USD. "
        f"{flags_sentence} "
        "Based on the applicable anti-money laundering policies in the knowledge base, "
        "what actions and best practices should be considered for this client and this "
        "transaction? Please:\n"
        "- Identify which AML policies (including any related to PEP, sanctions, FATF, or OFAC) are relevant.\n"
        "- Describe the appropriate investigative steps and documentation.\n"
        "- Recommend any enhanced due diligence, monitoring, escalation, or reporting that may be required.\n"
        "- If no additional actions are required, explicitly state that and explain why."
    )

    return query


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def _build_main_agent() -> agents.Agent:
    """Construct the main AML ReAct agent."""
    return agents.Agent(
        name="Anti Money Laundering Agent",
        instructions=REACT_INSTRUCTIONS,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
    )


async def run_aml_query(question: str) -> str:
    """
    Run the AML ReAct agent on a single question and return the final assistant answer text.
    """
    main_agent = _build_main_agent()

    # Use the same streaming mechanism as the Gradio demo, but capture the messages.
    result_stream = agents.Runner.run_streamed(main_agent, input=question)

    gr_messages: List[ChatMessage] = []

    async for event in result_stream.stream_events():
        gr_messages += oai_agent_stream_to_gradio_messages(event)

    # Extract the latest assistant message
    final_answer = ""
    for msg in reversed(gr_messages):
        # ChatMessage is typically (role, content)
        role = getattr(msg, "role", None) or getattr(msg, "sender", None)
        content = getattr(msg, "content", None)
        if role == "assistant" and content:
            final_answer = content
            break

    return final_answer.strip()


async def run_aml_for_row_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper: take a raw CSV row (dict), build query, run agent,
    and return extended row with 'aml_query' and 'aml_answer'.
    """
    aml_row = _row_from_dict(row)
    query = build_aml_query(aml_row)
    answer = await run_aml_query(query)

    extended_row: Dict[str, Any] = dict(row)
    extended_row["aml_query"] = query
    extended_row["aml_answer"] = answer
    return extended_row


# ---------------------------------------------------------------------------
# Batch CSV processing
# ---------------------------------------------------------------------------

async def process_csv(input_path: str, output_path: Optional[str] = None) -> None:
    """
    Batch process all records in a CSV through the AML agent.

    Writes original columns + aml_query + aml_answer to output_path if provided,
    otherwise prints results to stdout.
    """
    logging.info("Reading CSV from %s", input_path)

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logging.warning("No rows found in CSV.")
        return

    # Process rows sequentially (you can make this concurrent later if you want)
    processed_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        logging.info("Processing row %d / %d (client_id=%s)", idx, len(rows), row.get("client_id"))
        extended = await run_aml_for_row_dict(row)
        processed_rows.append(extended)

    if output_path:
        fieldnames = list(rows[0].keys()) + ["aml_query", "aml_answer"]
        logging.info("Writing results to %s", output_path)
        with open(output_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for r in processed_rows:
                writer.writerow(r)
    else:
        # No output file => just dump to stdout in a readable way.
        for r in processed_rows:
            print("=" * 80)
            print(f"client_id: {r.get('client_id')}, client_name: {r.get('client_name')}")
            print(f"aml_query:\n{r['aml_query']}\n")
            print("aml_answer:")
            print(r["aml_answer"])
            print()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AML ReAct RAG CLI – single-record and batch CSV processing."
    )

    # Batch mode
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to input CSV file (batch mode).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to output CSV file with aml_query and aml_answer columns.",
    )

    # Single-record mode
    parser.add_argument("--client-id", type=str, help="Client ID (single-record mode).")
    parser.add_argument("--client-name", type=str, help="Client name (single-record mode).")
    parser.add_argument("--pep-flag", type=int, help="PEP flag (0 or 1).")
    parser.add_argument("--sanctions-flag", type=int, help="Sanctions flag (0 or 1).")
    parser.add_argument("--fatf-country-flag", type=int, help="FATF country flag (0 or 1).")
    parser.add_argument("--ofac-country-flag", type=int, help="OFAC country flag (0 or 1).")
    parser.add_argument("--transaction-amount", type=float, help="Transaction amount.")

    args = parser.parse_args(argv)

    # Validate mode
    if args.csv:
        # batch mode, nothing else strictly required
        return args

    # single-record mode: make sure all fields present
    required_single = [
        "client_id",
        "client_name",
        "pep_flag",
        "sanctions_flag",
        "fatf_country_flag",
        "ofac_country_flag",
        "transaction_amount",
    ]
    missing = [f for f in required_single if getattr(args, f.replace("-", "_"), None) is None]

    # argparse converts "--pep-flag" to args.pep_flag etc.
    # We had names without hyphens in add_argument, so we can simply use attrs as defined.

    missing = [
        name
        for name in [
            "client_id",
            "client_name",
            "pep_flag",
            "sanctions_flag",
            "fatf_country_flag",
            "ofac_country_flag",
            "transaction_amount",
        ]
        if getattr(args, name, None) is None
    ]

    if missing:
        parser.error(
            "Single-record mode: you must provide all of "
            "--client-id, --client-name, --pep-flag, --sanctions-flag, "
            "--fatf-country-flag, --ofac-country-flag, --transaction-amount "
            "when --csv is not specified."
        )

    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    try:
        if args.csv:
            # Batch mode
            asyncio.run(process_csv(args.csv, args.output))
        else:
            # Single-record mode
            row = {
                "client_id": args.client_id,
                "client_name": args.client_name,
                "pep_flag": str(args.pep_flag),
                "sanctions_flag": str(args.sanctions_flag),
                "fatf_country_flag": str(args.fatf_country_flag),
                "ofac_country_flag": str(args.ofac_country_flag),
                "transaction_amount": str(args.transaction_amount),
            }
            aml_row = _row_from_dict(row)
            query = build_aml_query(aml_row)
            answer = asyncio.run(run_aml_query(query))

            print("=== AML QUERY ===")
            print(query)
            print("\n=== AML ANSWER ===")
            print(answer)
    finally:
        asyncio.run(_cleanup_clients())


if __name__ == "__main__":
    main()
