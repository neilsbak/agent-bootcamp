from openai import OpenAI
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import pandas as pd


from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
)

load_dotenv(verbose=True)

client = OpenAI()

def load_kyc_dataset(client_id: int) -> pd.DataFrame:
    ##Synthetic Data True Positives 
    SPREADSHEET_ID = '13DIJXTjGcm34Xf6e7APsGd1hkNc_hfSTIwNCRhF6tNg' 
    GID_clientdata = '466321727'
    GID_transaction = '681908835' 
    url_clientdata = f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?gid={GID_clientdata}&format=csv'
    url_transaction = f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?gid={GID_transaction}&format=csv'

    try:
        df_clientdata = pd.read_csv(url_clientdata)
    except Exception as e:
        print(f"Error reading public sheet: {e}")
        
    try:
        df_transaction = pd.read_csv(url_transaction)
    except Exception as e:
        print(f"Error reading public sheet: {e}")

    # Merge datasets on 'client_id'
    ###df_kyc = pd.merge(df_transaction, df_clientdata, on='client_id', how='left')
    df_clientdata = df_clientdata.where(df_clientdata['client_id'] == client_id).dropna()
    df_transaction = df_transaction.where(df_transaction['client_id'] == client_id).dropna()
    return df_clientdata, df_transaction


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

    df_client, df_transaction = load_kyc_dataset(txn.get("client_id"))

    kyc = {
        "client_id": df_client.get("client_id").values[0],
        "client_name": df_client.get("client_name").values[0],
        "country": df_client.get("client_country").values[0],
        "flag_high_risk_country": (df_client.get("fatf_country_flag").values[0]==1)|(df_client.get("ofac_country_flag").values[0]==1),
        "flag_pep": df_client.get("pep_flag").values[0],
        "flag_high_net_worth": df_client.get("HNW").values[0],
        "income": df_client.get("Income").values[0],
        "Deposit_3MonthsAvg": df_client.get("Deposit (3 months moving average)").values[0]
        
    }

    return {"kyc": kyc}


def node_websearch(state: AMLState) -> Dict[str, Any]:
    kyc = state["kyc"]

    websearch = {
        
    }

    return {"websearch": websearch}


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
        "customer_id": "C112"
    },
    "kyc": {},
    "websearch": {},
    "risk_score": 0.0,
    "alerts": [],
    "policycheck": {},
    "explanation": ""
}

result = aml_graph.invoke(flagged_txn)
print(result)