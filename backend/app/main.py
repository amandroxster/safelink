from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
import boto3
import json
import os
import logging
import traceback

# ===== Logging Setup =====
logger = logging.getLogger("SafeLinkAgentCore")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = True
logger.info("✅ Logging configured successfully")

# ===== FastAPI App =====
app = FastAPI(title="SafeLink Agent Core - AWS Bedrock AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== AWS Bedrock Client =====
REGION = os.getenv("AWS_REGION", "us-east-2")
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.meta.llama3-1-70b-instruct-v1:0"  # keep your working model
)
BEDROCK = boto3.client("bedrock-runtime", region_name=REGION)
logger.info("✅ Bedrock client initialized in %s", REGION)

# ===== Input Schema =====
class IncidentReport(BaseModel):
    message: str

# ===== In-memory incident queue =====
INCIDENT_QUEUE = []

# ===== Bedrock Helper =====
def call_bedrock(prompt: str) -> str:
    """
    Calls AWS Bedrock using the inference profile.
    Avoids using max_tokens; prompt instructs model to limit output.
    Trims output to 500 chars to avoid huge responses.
    """
    logger.info("➡️ Sending prompt to Bedrock: %s", prompt)
    try:
        body = json.dumps({"prompt": prompt})
        response = BEDROCK.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        raw_output = json.loads(response["body"].read())
        output_text = raw_output.get("results", [{}])[0].get("outputText", "")
        if len(output_text) > 500:
            output_text = output_text[:500] + "..."
        logger.info("⬅️ Bedrock response: %s", output_text)
        return output_text.strip()
    except Exception as e:
        logger.error("❌ Bedrock call failed: %s", e)
        logger.debug(traceback.format_exc())
        return "Error: Unable to process request"

# ===== Tools =====
def severity_tool(message: str) -> str:
    prompt = (
        f"Classify the severity of this incident in ONE word (High, Medium, Low) ONLY: {message}"
    )
    result = call_bedrock(prompt)
    logger.info("Severity result: %s", result)
    return result

def summarization_tool(message: str) -> str:
    prompt = (
        f"Summarize this incident in ONE short sentence, concise for responders: {message}"
    )
    result = call_bedrock(prompt)
    logger.info("Summary result: %s", result)
    return result

def citizen_guidance_tool(message: str) -> str:
    prompt = (
        f"Provide very brief citizen safety guidance in 1-2 sentences: {message}"
    )
    result = call_bedrock(prompt)
    logger.info("Guidance result: %s", result)
    return result

# ===== API Routes =====
@app.post("/incident")
def handle_incident(report: IncidentReport):
    logger.info("Received incident report: %s", report.message)
    severity = severity_tool(report.message)
    summary = summarization_tool(report.message)
    guidance = citizen_guidance_tool(report.message)

    incident = {
        "severity": severity,
        "responder_summary": summary,
        "citizen_guidance": guidance
    }
    INCIDENT_QUEUE.append(incident)
    logger.info("Incident processed and added to queue: %s", incident)
    return incident

@app.get("/incidents")
def get_incidents():
    logger.info("Fetching all incidents, count: %d", len(INCIDENT_QUEUE))
    for i, incident in enumerate(INCIDENT_QUEUE, start=1):
        logger.info("Incident %d: %s", i, incident)
    return INCIDENT_QUEUE

@app.get("/")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "SafeLink Agent Core is running"}

# ===== Bedrock Test Endpoint =====
@app.post("/bedrock-test")
def bedrock_test(payload: dict):
    prompt = payload.get("prompt", "Say hello from Bedrock!")
    try:
        text = call_bedrock(prompt)
        return {"bedrock_text": text}
    except Exception as e:
        logger.error("❌ Bedrock test failed: %s", e)
        return {"error": str(e)}

# ===== Lambda Handler =====
handler = Mangum(app)
logger.info("Mangum handler initialized")
logger.info("✅ FastAPI app initialized successfully")