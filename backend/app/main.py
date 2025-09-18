from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import json
import os
import logging
from mangum import Mangum

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== FastAPI App =====
app = FastAPI(title="SafeLink Agent Core - AWS Titan AI")

# ===== AWS Bedrock Client =====
region = os.getenv("AWS_REGION", "us-east-2")
session = boto3.Session(region_name=region)
bedrock = session.client("bedrock-runtime", region_name=region)

# Use publicly available Titan model
MODEL_ID = "amazon.titan-text-1"

# ===== Input Schema =====
class IncidentReport(BaseModel):
    message: str

# ===== In-memory incident queue (POC) =====
INCIDENT_QUEUE = []

# ===== Bedrock Call Helper =====
def call_bedrock(prompt: str) -> str:
    try:
        body = json.dumps({
            "text": prompt,  # Titan expects 'text' key
            "max_tokens": 150,
            "temperature": 0.3
        })
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        output = json.loads(response["body"].read())
        # Titan returns generated text in 'outputText'
        return output.get("outputText", "").strip()
    except Exception as e:
        logger.error("Bedrock call failed: %s", e)
        return "Error: Unable to process request"

# ===== Tools =====
def severity_tool(message: str) -> str:
    prompt = f"Classify the severity of this incident (High, Medium, Low): {message}"
    return call_bedrock(prompt)

def summarization_tool(message: str) -> str:
    prompt = f"Summarize this incident for responders: {message}"
    return call_bedrock(prompt)

def citizen_guidance_tool(message: str) -> str:
    prompt = f"Provide citizen safety guidance for this incident: {message}"
    return call_bedrock(prompt)

# ===== API Routes =====
@app.post("/incident")
def handle_incident(report: IncidentReport):
    severity = severity_tool(report.message)
    summary = summarization_tool(report.message)
    guidance = citizen_guidance_tool(report.message)

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

# ===== Mangum Handler for Lambda =====
handler = Mangum(app)
