from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import json
import os

app = FastAPI(title="SafeLink Agent Core")

# ===== AWS Bedrock Client =====
session = boto3.Session(region_name=os.getenv("AWS_REGION", "us-east-1"))
bedrock = session.client("bedrock-runtime")

# ===== Input Schema =====
class IncidentReport(BaseModel):
    message: str

# ===== In-Memory Incident Queue for POC =====
# For demo only, in real deployment use a DB or queue service
INCIDENT_QUEUE = []

# ===== Bedrock Call Helper =====
def call_bedrock(prompt: str) -> str:
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

# ===== Tools =====
def severity_tool(message: str) -> str:
    prompt = f"Classify the severity of this emergency as High, Medium, or Low:\nIncident: {message}"
    return call_bedrock(prompt)

def summarization_tool(message: str) -> str:
    prompt = f"Summarize this emergency for first responders in one concise sentence:\nIncident: {message}"
    return call_bedrock(prompt)

def citizen_guidance_tool(message: str) -> str:
    prompt = f"Provide clear safety instructions for a citizen facing this emergency:\nIncident: {message}"
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

    # Append to in-memory queue for Responder view
    INCIDENT_QUEUE.append(incident)
    return incident

@app.get("/incidents")
def get_incidents():
    # Return all incidents (Responder dashboard)
    return INCIDENT_QUEUE

@app.get("/")
def health_check():
    return {"status": "SafeLink Agent Core is running"}
