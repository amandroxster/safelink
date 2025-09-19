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
    allow_origins=["*"],  # dev only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== AWS Bedrock Client =====
REGION = os.getenv("AWS_REGION", "us-east-2")
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.meta.llama3-1-70b-instruct-v1:0"  # your working model
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
    Calls AWS Bedrock Llama3-Instruct model.
    Uses the correct key 'generation' for output text.
    Ensures short responses via prompt instructions.
    Trims output to 500 chars max.
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
        logger.info("Full raw Bedrock response: %s", raw_output)

        # Correct extraction
        output_text = raw_output.get("generation", "").strip()
        if not output_text:
            logger.warning("Bedrock returned empty generation")
            output_text = "No response from model."

        # Trim excessively long outputs
        if len(output_text) > 500:
            output_text = output_text[:500] + "..."

        logger.info("⬅️ Bedrock response (trimmed): %s", output_text)
        return output_text

    except Exception as e:
        logger.error("❌ Bedrock call failed: %s", e)
        logger.debug(traceback.format_exc())
        return "Error: Unable to process request"

# ===== Tools =====
def severity_tool(message: str) -> str:
    prompt = f"Classify the severity of this incident in ONE word (High, Medium, Low) ONLY: {message}"
    return call_bedrock(prompt)

def summarization_tool(message: str) -> str:
    prompt = f"Summarize this incident in ONE short sentence, concise for responders: {message}"
    return call_bedrock(prompt)

def citizen_guidance_tool(message: str) -> str:
    prompt = f"Provide very brief citizen safety guidance in 1-2 sentences: {message}"
    return call_bedrock(prompt)

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
    return INCIDENT_QUEUE

@app.get("/")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "SafeLink Agent Core is running"}

@app.post("/bedrock-test")
def bedrock_test(payload: dict):
    """
    Simple Bedrock test endpoint.
    Expects JSON: {"prompt": "Your prompt here"}
    """
    prompt = payload.get(
        "prompt",
        "Provide one short, concise safety tip for a fire incident in under 30 words."
    )
    try:
        text = call_bedrock(prompt)
        return {"bedrock_text": text}
    except Exception as e:
        logger.error("❌ Bedrock test failed: %s", e)
        return {"error": str(e)}

# ===== Lambda Handler =====
handler = Mangum(app)
logger.info("Mangum handler initialized")
logger.info("🚀 FastAPI app is ready to serve requests")
