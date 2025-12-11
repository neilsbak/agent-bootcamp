import asyncio

from openai import OpenAI
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END

from TD_One.kyc_web_search import run_kyc_web_search

load_dotenv(verbose=True)

client = OpenAI()

####### Define State - records updates at each node ######
class AMLState(TypedDict):
    transaction: Dict[str, Any]
    kyc: Dict[str, Any]
    websearch: Dict[str, Any]
    risk_score: float
    alerts: List[str]
    policycheck: Dict[str, Any]
    explanation: str



####### Node 1: KYC #######
def node_kyc(state: AMLState) -> Dict[str, Any]:
    txn = state["transaction"]

    kyc = {
        "high_risk_country": txn.get("country") in ["AB", "CD", "EF"],
        "amount_usd": float(txn.get("amount", 0)),
        "customer_risk_segment": "HNW" if txn.get("amount", 0) > 10000 else "Regular"
    }

    return {"kyc": kyc}


async def node_websearch(state: AMLState) -> Dict[str, Any]:
    """Run KYC web search on the customer."""
    txn = state["transaction"]
    customer_name = txn.get("customer_name", "")

    if not customer_name:
        return {"websearch": {"error": "No customer name provided", "results": ""}}

    search_results = await run_kyc_web_search(customer_name)

    return {"websearch": {"results": search_results}}


def node_risk_score(state: AMLState) -> Dict[str, Any]:
    kyc = state["kyc"]

    score = 0
    if kyc.get("high_risk_country"):
        score += 60
    if kyc.get("amount_usd", 0) > 10000:
        score += 30
    if kyc.get("customer_risk_segment") == "HNW":
        score += 10

    return {"risk_score": score}


def node_generate_alerts(state: AMLState) -> Dict[str, Any]:
    score = state["risk_score"]

    if score > 80:
        alerts = ["HIGH-RISK AML ALERT: escalate immediately"]
    elif score > 40:
        alerts = ["Medium risk: manual review recommended"]
    else:
        alerts = ["Low AML risk"]

    return {"alerts": alerts}



def node_policycheck(state: AMLState) -> Dict[str, Any]:
    kyc = state["kyc"]
    websearch = state["websearch"]

    policycheck = {

    }

    return {"policycheck": policycheck}


def node_explain_output(state: AMLState) -> Dict[str, Any]:
    score = state["risk_score"]
    alerts = state["alerts"]

    prompt = f"""
    Provide a concise AML explanation for the following:
    Risk Score: {score}
    Alerts: {alerts}
    """

    MAX_TURNS = 5
    AGENT_LLM_NAME = "gemini-2.5-flash"
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        #max_tokens=2000,
        #temperature=0
    )

    explanation = response.choices[0].message.content.strip()

    return {"explanation": explanation}


graph = StateGraph(AMLState)

graph.add_node("kyc", node_kyc)
graph.add_node("websearch", node_websearch)
graph.add_node("score", node_risk_score)
graph.add_node("alert", node_generate_alerts)
graph.add_node("policycheck", node_policycheck)
graph.add_node("explain", node_explain_output)

graph.set_entry_point("kyc")
graph.add_edge("kyc", "websearch")
graph.add_edge("websearch", "score")
graph.add_edge("score", "alert")
graph.add_edge("alert", "policycheck")
graph.add_edge("policycheck", "explain")
graph.add_edge("explain", END)

aml_graph = graph.compile()


flagged_txn = {
    "transaction": {
        "txn_id": "T9912",
        "amount": 15000,
        "country": "AE",
        "customer_id": "C112",
        "customer_name": "John Smith, CEO of Acme Corp",
    },
    "kyc": {},
    "websearch": {},
    "risk_score": 0.0,
    "alerts": [],
    "policycheck": {},
    "explanation": ""
}


async def main():
    result = await aml_graph.ainvoke(flagged_txn)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
