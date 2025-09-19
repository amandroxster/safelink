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
    allow_origins=["*"],  # dev only, lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== AWS Bedrock Client =====
REGION = os.getenv("AWS_REGION", "us-east-2")
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.meta.llama3-1-70b-instruct-v1:0"  # keep the working model
)
BEDROCK = boto3.client("bedrock-runtime", region_name=REGION)
logger.info("✅ Bedrock client initialized in %s", REGION)

# ===== Input Schema =====
class IncidentReport(BaseModel):
    message: str

# ===== In-memory incident queue =====
INCIDENT_QUEUE = []

# ===== Bedrock Helper =====
def call_bedrock(prompt: str, max_tokens: int = 100) -> str:
    """
    Calls AWS Bedrock using the inference profile.
    Limits output length to prevent massive responses.
    """
    try:
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.3
        })
        response = BEDROCK.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        raw_output = json.loads(response["body"].read())
        # extract outputText if present
        output_text = raw_output.get("results", [{}])[0].get("outputText", "")
        if len(output_text) > 500:
            output_text = output_text[:500] + "..."
        return output_text.strip()
    except Exception as e:
        logger.error("❌ Bedrock call failed: %s", e)
        logger.debug(traceback.format_exc())
        return "Error: Unable to process request"

# ===== Tools =====
def severity_tool(message: str) -> str:
    return call_bedrock(f"Classify the severity (High, Medium, Low) of this incident: {message}", max_tokens=10)

def summarization_tool(message: str) -> str:
    return call_bedrock(f"Summarize this incident in one short sentence: {message}", max_tokens=50)

def citizen_guidance_tool(message: str) -> str:
    return call_bedrock(f"Provide concise citizen safety guidance for this incident: {message}", max_tokens=50)

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
    logger.info("Incident processed: %s", incident)
    return incident

@app.get("/incidents")
def get_incidents():
    return INCIDENT_QUEUE

@app.get("/")
def health_check():
    return {"status": "SafeLink Agent Core is running"}

# ===== Bedrock Test Endpoint =====
@app.post("/bedrock-test")
def bedrock_test(payload: dict):
    prompt = payload.get("prompt", "Say hello from Bedrock!")
    logger.info("➡️ Sending prompt to Bedrock: %s", prompt)
    try:
        text = call_bedrock(prompt, max_tokens=100)
        logger.info("⬅️ Bedrock response: %s", text)
        return {"bedrock_text": text}
    except Exception as e:
        logger.error("❌ Bedrock test failed: %s", e)
        return {"error": str(e)}

# ===== Lambda Handler =====
handler = Mangum(app)
logger.info("Mangum handler initialized")
logger.info("🚀 FastAPI app is ready to handle requests")