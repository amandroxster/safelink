from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import boto3
import os
import json
import logging
from mangum import Mangum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SafeLink Agent Core - Bedrock POC")

# ===== CORS Middleware =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Amplify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== AWS Bedrock Client =====
region = os.getenv("AWS_REGION", "us-east-2")
bedrock = boto3.client("bedrock-runtime", region_name=region)

# ===== Input Schema =====
class IncidentReport(BaseModel):
    message: str

# ===== In-Memory Incident Queue =====
INCIDENT_QUEUE = []

# ===== Bedrock Helpers =====
def get_first_available_model() -> str:
    try:
        response = bedrock.list_models()
        for model in response.get("modelSummaries", []):
            model_id = model["modelId"]
            # Use the first Anthropic or Claude model available
            if "anthropic" in model_id.lower() or "claude" in model_id.lower():
                return model_id
        return None
    except Exception as e:
        logger.error("Failed to list Bedrock models: %s", e)
        return None

def call_bedrock(prompt: str) -> str:
    model_id = get_first_available_model()
    if not model_id:
        logger.warning("No Bedrock model found, using fallback response.")
        return "Error: Unable to process request"

    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 100,
        "temperature": 0.3
    })

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
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

# ===== Mangum handler for Lambda =====
handler = Mangum(app)
