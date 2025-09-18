from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import os
import logging
from mangum import Mangum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SafeLink Agent Core - AWS Native")

# Initialize AWS clients
region = os.getenv("AWS_REGION", "us-east-2")
comprehend = boto3.client("comprehend", region_name=region)

# Input schema
class IncidentReport(BaseModel):
    message: str

# In-memory incident queue for POC
INCIDENT_QUEUE = []

# ===== Tools using AWS Comprehend =====
def severity_tool(message: str) -> str:
    """
    Classify severity based on keywords or sentiment.
    Simple POC logic:
      - HIGH: contains fire, flood, accident, critical
      - MEDIUM: minor injury, small fire, moderate
      - LOW: noise complaint, lost item
    """
    keywords_high = ["fire", "flood", "accident", "critical", "injury"]
    keywords_medium = ["minor", "moderate", "small"]
    
    msg_lower = message.lower()
    if any(word in msg_lower for word in keywords_high):
        return "High"
    elif any(word in msg_lower for word in keywords_medium):
        return "Medium"
    else:
        return "Low"

def summarization_tool(message: str) -> str:
    """
    Simple summarization using Comprehend key phrases.
    """
    try:
        resp = comprehend.detect_key_phrases(Text=message, LanguageCode="en")
        phrases = [kp["Text"] for kp in resp.get("KeyPhrases", [])]
        summary = ", ".join(phrases[:5])
        return f"Summary: {summary}" if summary else message[:50]
    except Exception as e:
        logger.error("Comprehend summarization failed: %s", e)
        return message[:50]

def citizen_guidance_tool(message: str) -> str:
    """
    Generate basic citizen guidance based on severity.
    """
    severity = severity_tool(message)
    if severity == "High":
        return "Evacuate immediately and call 911."
    elif severity == "Medium":
        return "Be cautious and monitor updates from authorities."
    else:
        return "Stay alert, no immediate danger detected."

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

# ===== Mangum handler for AWS Lambda =====
handler = Mangum(app)
