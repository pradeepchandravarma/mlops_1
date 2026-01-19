from sqlalchemy import create_engine
from sqlalchemy import text
import boto3
import json
from botocore.exceptions import ClientError
import pandas as pd
from dotenv import *

def get_secret():
    secret_name = "pradeep-mlops-postgres"
    region_name = "eu-west-2"

    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name
    )

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    return json.loads(response["SecretString"])
# Database Access Layer
def get_engine():
    secret = get_secret()
    return create_engine(
        f"postgresql+psycopg2://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )


#Safe SQL query function (IMPORTANT)
def query_db(sql: str) -> pd.DataFrame:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

#Load ML Model (Once)
import joblib

model = joblib.load("models/student_performance_sgd.joblib")

#Preprocessing Logic (CRITICAL)
def preprocess_features(features: dict) -> pd.DataFrame:
    data = features.copy()

    # Convert Yes/No → 1/0
    data["Extracurricular Activities"] = (
        1 if data["Extracurricular Activities"].lower() == "yes" else 0
    )

    df = pd.DataFrame([data])

    # Ensure correct column order (VERY IMPORTANT)
    df = df[
        [
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced"
        ]
    ]

    return df
#Prediction Tool
def predict_performance(features: dict):
    df = preprocess_features(features)
    prediction = model.predict(df)
    return float(prediction[0])
#LLM Agent Prompt (THIS MAKES OR BREAKS IT)

SYSTEM_PROMPT = """
You are an intelligent MLOps assistant.

You have two tools:
1. Query a PostgreSQL table named mlops.performance_metrics
2. Predict student performance using a trained ML model

Rules:
- Use SQL for historical data, statistics, filtering, aggregation
- Use prediction tool only when the user asks to predict, estimate, or forecast
- SQL queries must be SELECT only
- The prediction output is Performance Index

The table mlops.performance_metrics has columns:
- hours_studied (integer)
- previous_scores (integer)
- extracurricular_activities (text: Yes/No)
- sleep_hours (integer)
- sample_question_papers_practiced (integer)
- performance_index (integer)
"""

#Tool Definitions (OpenAI-compatible)
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "Query performance metrics from PostgreSQL",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_performance",
            "description": "Predict performance index using ML model",
            "parameters": {
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "properties": {
                            "Hours Studied": {"type": "number"},
                            "Previous Scores": {"type": "number"},
                            "Extracurricular Activities": {"type": "string"},
                            "Sleep Hours": {"type": "number"},
                            "Sample Question Papers Practiced": {"type": "number"}
                        },
                        "required": [
                            "Hours Studied",
                            "Previous Scores",
                            "Extracurricular Activities",
                            "Sleep Hours",
                            "Sample Question Papers Practiced"
                        ]
                    }
                },
                "required": ["features"]
            }
        }
    }
]
#Agent Execution Loop
from openai import OpenAI
import json

client = OpenAI()

def run_agent(user_question: str):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ],
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "query_db":
            df = query_db(args["sql"])
            return df.to_string(index=False)

        if name == "predict_performance":
            pred = predict_performance(args["features"])
            return f"Predicted Performance Index: {pred:.2f}"

    return msg.content
