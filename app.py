import os
import streamlit as st
from google.cloud import bigquery
from agents import Agent, Runner, function_tool, ItemHelpers
from dataclasses import dataclass
from typing import List
from openai import OpenAI
import asyncio
import tempfile
import json

# ----------------------------
# Setup
# ----------------------------
gcp_credentials = dict(st.secrets["gcp_service_account"])

# Write them to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
    f.write(json.dumps(gcp_credentials).encode("utf-8"))
    f.flush()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name

openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

bq_client = bigquery.Client()
llm_client = OpenAI(api_key=openai_api_key)

# ----------------------------
# Optimized BigQuery Functions
# ----------------------------

@function_tool
def get_high_basket_value_orders(amount: int):
    """
    Fetch orders with basket value greater than the specified amount. 
    """
    query = f"""
    SELECT  
        sales_fact.order_id,
        sales_fact.customer_sk,
        sales_fact.quantity,
        product_dim.product_name,
        product_dim.purchase_price,
        (sales_fact.quantity * product_dim.purchase_price) AS basket_value
    FROM `greensupplyproject.retail_data.Fact_Sales` AS sales_fact
    INNER JOIN `greensupplyproject.retail_data.Dim_Product` AS product_dim 
        ON sales_fact.product_sk = product_dim.product_sk
    WHERE (sales_fact.quantity * product_dim.purchase_price) > {amount}
    ORDER BY basket_value DESC
    LIMIT 50
    """
    df = bq_client.query(query).to_dataframe()
    return df.to_dict(orient="records")

@function_tool
def get_customer_names_in_batch(customer_sks: List[str]):
    """
    Fetch multiple customer names in one batch query using IN clause.
    Returns a mapping of {customer_sk: company_name}.
    """
    if not customer_sks:
        return {}
    formatted_sks = ','.join([f"'{sk}'" for sk in customer_sks])
    query = f"""
    SELECT customer_sk, company_name
    FROM `greensupplyproject.retail_data.Dim_Customer`
    WHERE customer_sk IN ({formatted_sks})
    """
    df = bq_client.query(query).to_dataframe()
    return dict(zip(df["customer_sk"], df["company_name"]))

# ----------------------------
# Agent Definition
# ----------------------------
@dataclass
class AgentBasket:
    order_id: str
    company_name: str
    basket_value: float
    issues: List[str]


shopping_cart_anomaly_agent = Agent(
    name="ShoppingCartAnomalyAgent",
    instructions="""
    You are a data analysis agent that identifies unusually high basket values in retail transactions.

    ### RULES ###
    1. Call get_high_basket_value_orders(amount) as the first step using the numeric threshold the user provides.
    2. Extract all unique customer_sks from the results and call get_customer_names_in_batch once.
    3. For each order:
       - Find the company_name from the batch result (use 'UNKNOWN' if missing).
       - Create one JSON object with:
           "order_id", "company_name", and "issues" (a short explanation).
    4. Output a JSON array exactly matching:
       [
         {
           "order_id": "<string>",
           "company_name": "<string>",
           "basket_value": <float>,
           "issues": ["<short explanation>"]
         }
       ]
    5. Keep explanations short, like:
       "Basket value $1,250 exceeds threshold; high quantity * high price."
    6. If no high-value orders exist, return [].
    7. Return only the JSON (no extra text).
    """,
    tools=[get_high_basket_value_orders, get_customer_names_in_batch],
    output_type=List[AgentBasket],
    model="gpt-4o-mini"
)

# ----------------------------
# Runner
# ----------------------------
st.title("Agent Basket Report")

st.markdown("Provide the agent an basket value threshold")
user_content = st.text_input("Agent input", value="detect unusual basket values greater than 450.")
run_btn = st.button("Run Detection")

if run_btn:
    runner = Runner()
    input_message = [{"role": "user", "content": user_content}]
    try:
        with st.spinner("Running agent..."):
            result = asyncio.run(runner.run(shopping_cart_anomaly_agent, input_message, max_turns=10))
        output = ItemHelpers.text_message_outputs(result.new_items)
        if not output or output == "[]":
            st.info("No high-value orders found.")
        else:
            st.subheader("Generated Anomalies")
            # If output is JSON text, display as JSON; otherwise show raw
            try:
                import json
                parsed = json.loads(output)
                st.json(parsed)
            except Exception:
                st.write(output)
    except Exception as e:
        st.error(f"Error running agent: {e}")
# ...existing code...