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
REGION = os.getenv("AWS_REGION", "us-east-2")
SESSION = boto3.Session(region_name=REGION)
BEDROCK = SESSION.client("bedrock-runtime", region_name=REGION)
BEDROCK_API = SESSION.client("bedrock", region_name=REGION)

# ===== Inference Profile Config =====
# Set the profile ARN via environment variable
INFERENCE_PROFILE_ARN = os.getenv(
    "BEDROCK_INFERENCE_PROFILE_ARN",
    "arn:aws:bedrock:us-east-2:158491568534:inference-profile/us.meta.llama3-2-1b-instruct-v1:0"
)

# ===== Input Schema =====
class IncidentReport(BaseModel):
    message: str

# ===== In-memory incident queue (POC) =====
INCIDENT_QUEUE = []

# ===== Helper: Get model ARN from inference profile =====
def get_model_arn(profile_arn: str, region: str) -> str:
    """
    Returns the model ARN from the inference profile that matches the current region.
    """
    try:
        # List all inference profiles
        response = BEDROCK_API.list_inference_profiles()
        profile = next(
            p for p in response["inferenceProfileSummaries"]
            if p["inferenceProfileArn"] == profile_arn
        )

        # Pick the first model ARN that contains the region string
        model_arn = next(
            m["modelArn"] for m in profile["models"] if region in m["modelArn"]
        )
        return model_arn
    except Exception as e:
        logger.error("Failed to get model ARN from inference profile: %s", e)
        raise

# Resolve MODEL_ID dynamically
MODEL_ID = get_model_arn(INFERENCE_PROFILE_ARN, REGION)

# ===== Bedrock Call Helper =====
def call_bedrock(prompt: str) -> str:
    try:
        body = json.dumps({
            "text": prompt,  # Titan/Nova Lite expects 'text' key
            "max_tokens": 150,
            "temperature": 0.3
        })
        response = BEDROCK.invoke_model(
            modelId=MODEL_ID,
            inferenceProfileArn=INFERENCE_PROFILE_ARN,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        output = json.loads(response["body"].read())
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
