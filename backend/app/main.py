from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json
import os
import logging
from mangum import Mangum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SafeLink Agent Core - Anthropic Claude 3.5 Sonnet v2")

# ===== CORS Middleware =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with Amplify domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== AWS Bedrock Client for Anthropic =====
region = os.getenv("AWS_REGION", "us-east-2")
session = boto3.Session(region_name=region)
bedrock = session.client("bedrock-runtime")

# ===== Input Schema =====
class IncidentReport(BaseModel):
    message: str

# ===== In-Memory Incident Queue =====
INCIDENT_QUEUE = []

# ===== AI Call Helper =====
def call_anthropic(prompt: str) -> str:
    """
    Calls Anthropic Claude 3.5 Sonnet v2 via Amazon Bedrock.
    """
    try:
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 100,
            "temperature": 0.3
        })
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3.5-sonnet-v2",  # Claude 3.5 Sonnet v2
            contentType="application/json",
            accept="application/json",
            body=body
        )
        model_output = json.loads(response["body"].read())
        return model_output.get("completion", "").strip()
    except Exception as e:
        logger.error("Anthropic call failed: %s", e)
        return "Error: Unable to process request"

# ===== Tools =====
def severity_tool(message: str) -> str:
    prompt = f"Classify the severity of this emergency as High, Medium, or Low:\nIncident: {message}"
    return call_anthropic(prompt)

def summarization_tool(message: str) -> str:
    prompt = f"Summarize this emergency for first responders in one concise sentence:\nIncident: {message}"
    return call_anthropic(prompt)

def citizen_guidance_tool(message: str) -> str:
    prompt = f"Provide clear safety instructions for a citizen facing this emergency:\nIncident: {message}"
    return call_anthropic(prompt)

# ===== API Routes =====
@app.post("/incident")
def handle_incident(report: IncidentReport):
    logger.info("Received incident: %s", report.message)
    
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

# ===== Mangum Handler for AWS Lambda =====
handler = Mangum(app)
