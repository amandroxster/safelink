from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import os
import logging
from mangum import Mangum
import traceback

# ===== Logging Configuration =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SafeLinkAgentCore")

# ===== FastAPI App =====
app = FastAPI(title="SafeLink Agent Core - AWS Bedrock AI")

# ===== CORS Configuration =====
origins = ["*"]  # Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== AWS Bedrock Client =====
REGION = os.getenv("AWS_REGION", "us-east-2")
SESSION = boto3.Session(region_name=REGION)
BEDROCK = SESSION.client("bedrock-runtime", region_name=REGION)

logger.info("AWS Bedrock client initialized in region: %s", REGION)
logger.info("AWS Bedrock client session: %s", SESSION)

# ===== Inference Profile Configuration =====
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.meta.llama3-1-70b-instruct-v1:0")
logger.info("Using Bedrock model ID: %s", MODEL_ID)

# ===== Input Schema =====
class IncidentReport(BaseModel):
    message: str

# ===== In-memory incident queue =====
INCIDENT_QUEUE = []

# ===== Bedrock Call Helper =====
def call_bedrock(prompt_text: str) -> str:
    """
    Calls AWS Bedrock using a Meta Llama3 inference profile.
    Handles extraction of response text from Bedrock output.
    """
    logger.info("Preparing Bedrock request for prompt: %s", prompt_text)
    try:
        body = json.dumps({"prompt": prompt_text})
        logger.debug("Bedrock request body: %s", body)

        response = BEDROCK.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        raw_output = json.loads(response["body"].read())
        logger.info("Received Bedrock raw output: %s", raw_output)

        content = ""
        if "results" in raw_output:
            content = raw_output["results"][0].get("outputText", "")
        elif "choices" in raw_output:
            try:
                content = raw_output["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                logger.warning("Unexpected Bedrock response structure: %s", e)

        logger.info("Extracted content from Bedrock: %s", content)
        return content.strip()

    except Exception as e:
        logger.error("Bedrock call failed: %s", e)
        logger.debug(traceback.format_exc())
        return "Error: Unable to process request"

# ===== Tools =====
def severity_tool(message: str) -> str:
    logger.info("Running severity tool for message: %s", message)
    return call_bedrock(
        f"You are an AI assistant for incident management.\n"
        f"Classify the severity of this incident as High, Medium, or Low:\nIncident: {message}"
    )

def summarization_tool(message: str) -> str:
    logger.info("Running summarization tool for message: %s", message)
    return call_bedrock(
        f"You are an AI assistant for incident management.\n"
        f"Summarize this incident in one concise sentence for responders:\nIncident: {message}"
    )

def citizen_guidance_tool(message: str) -> str:
    logger.info("Running citizen guidance tool for message: %s", message)
    return call_bedrock(
        f"You are an AI assistant for incident management.\n"
        f"Provide safety guidance for citizens regarding this incident:\nIncident: {message}"
    )

# ===== API Routes =====
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Incoming request: %s %s", request.method, request.url)
    try:
        body = await request.body()
        if body:
            logger.debug("Request body: %s", body.decode())
    except Exception:
        logger.warning("Could not read request body")

    response = await call_next(request)
    logger.info("Response status code: %s", response.status_code)
    return response

@app.post("/incident")
def handle_incident(report: IncidentReport):
    logger.info("Received incident report: %s", report)
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

# ===== Mangum Handler for Lambda =====
handler = Mangum(app)
logger.info("Mangum handler initialized for AWS Lambda")
