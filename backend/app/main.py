from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import os
import json
import logging
from mangum import Mangum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SafeLink Agent Core")

session = boto3.Session(region_name=os.getenv("AWS_REGION", "us-east-2"))
bedrock = session.client("bedrock-runtime")

class IncidentReport(BaseModel):
    message: str

INCIDENT_QUEUE = []

def call_bedrock(prompt: str) -> str:
    try:
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 100,
            "temperature": 0.3
        })
        response = bedrock.invoke_model(
            modelId="anthropic.claude-v2",
            contentType="application/json",
            accept="application/json",
            body=body
        )
        model_output = json.loads(response["body"].read())
        return model_output.get("completion", "").strip()
    except Exception as e:
        logger.error("Bedrock call failed: %s", e)
        return "Error: Unable to process request"

@app.post("/incident")
def handle_incident(report: IncidentReport):
    logger.info("Received incident: %s", report.message)
    severity = call_bedrock(f"Classify severity: {report.message}")
    summary = call_bedrock(f"Summarize: {report.message}")
    guidance = call_bedrock(f"Citizen guidance: {report.message}")

    incident = {
        "severity": severity,
        "responder_summary": summary,
        "citizen_guidance": guidance
    }
    INCIDENT_QUEUE.append(incident)
    logger.info("Incident processed: %s", incident)
    return incident

@app.get("/incidents")
def get_incidents():
    return INCIDENT_QUEUE

@app.get("/")
def health_check():
    return {"status": "SafeLink Agent Core is running"}

# ===== Mangum handler for AWS Lambda =====
handler = Mangum(app)
