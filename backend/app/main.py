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

# Enable CORS (dev only; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SOS_PLAYBOOKS = {
    "Medical Emergency": [
        "Call 911 immediately via FirstNet.",
        "Provide exact location to dispatcher.",
        "Begin CPR if trained.",
        "Use AED if available and follow prompts.",
        "Stay with the person until responders arrive."
    ],
    "Hazmat": [
        "Move upwind, uphill, and upstream from the spill.",
        "Call 911 and notify HazMat response team.",
        "Isolate and deny entry to the affected area.",
        "Follow command center evacuation orders."
    ],
    "Active Shooter": [
        "Call 911 immediately via FirstNet.",
        "Follow 'Run, Hide, Fight' protocol.",
        "Provide dispatcher with number of shooters, location, and description.",
        "Do not approach law enforcement until cleared."
    ]
}

# ===== Logging Middleware for Debugging Paths =====
@app.middleware("http")
async def log_request_path(request: Request, call_next):
    logger.info("Incoming request path: %s", request.url.path)
    response = await call_next(request)
    return response

# ===== AWS Bedrock Client =====
REGION = os.getenv("AWS_REGION", "us-east-2")
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.meta.llama3-1-70b-instruct-v1:0"
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

        # Extract the generation output
        output_text = raw_output.get("generation", "").strip()
        if not output_text:
            logger.warning("Bedrock returned empty generation")
            output_text = "No response from model."

        # Trim output to 500 chars
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
    #guidance = citizen_guidance_tool(report.message)
    category = categorize_incident(report.message)
    print(f"category: {category}")
    sos_steps = SOS_PLAYBOOKS[category]
    if not sos_steps:
        sos_steps = "I am uncertain about the SOS steps. Please call 911"

    prompt = build_prompt(incident_text=report.message, category=category, sop_steps=sos_steps)

    guidance = call_bedrock(prompt)

    incident = {
        "severity": severity,
        "responder_summary": summary,
        "citizen_guidance": guidance
    }
    INCIDENT_QUEUE.append(incident)
    logger.info("Incident processed and added to queue: %s", incident)
    return incident


def categorize_incident(text: str) -> str:
    text_lower = text.lower()
    if "faint" in text_lower or "collapsed" in text_lower or "not breathing" in text_lower:
        return "Medical Emergency"
    elif "chemical" in text_lower or "spill" in text_lower:
        return "Hazmat"
    elif "shoot" in text_lower or "gun" in text_lower:
        return "Active Shooter"
    else:
        return "General Incident"  # fallback


def build_prompt(incident_text: str, category: str, sop_steps: list) -> str:
    sop_text = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(sop_steps)])
    prompt = f"""
    You are a FirstNet public safety assistant.
    Do NOT invent new steps. Use only the SOP steps provided below.

    Incident: "{incident_text}"
    Category: {category}

    SOP Steps:
    {sop_text}

    Format the response as a checklist for the responder.
    """
    return prompt

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
# ⚡ Key Fix: api_gateway_base_path trims /default/safelink-backend for FastAPI
handler = Mangum(app, api_gateway_base_path="/default/safelink-backend")
logger.info("Mangum handler initialized with base path")
logger.info("🚀 FastAPI app is ready to serve requests")
