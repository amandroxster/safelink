from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
app = FastAPI(title="SafeLink Agent Core - AWS Bedrock AI")

# ===== CORS Configuration =====
origins = [
    "*"  # Allow all origins for development; restrict in production
]
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

# ===== Inference Profile Configuration =====
# Use your inference profile ID from `aws bedrock list-inference-profiles`
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.meta.llama3-2-1b-instruct-v1:0")

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
    try:
        body = json.dumps({
            "prompt": prompt_text
        })

        response = BEDROCK.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        raw_output = json.loads(response["body"].read())
        logger.info("Bedrock raw output: %s", raw_output)

        # For Llama3, text may be under 'choices[0].message.content'
        content = ""
        if "choices" in raw_output:
            try:
                content = raw_output["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                logger.warning("Unexpected Bedrock response structure.")
        return content.strip()
    except Exception as e:
        logger.error("Bedrock call failed: %s", e)
        return "Error: Unable to process request"

# ===== Tools =====
def severity_tool(message: str) -> str:
    prompt = f"You are an AI assistant for incident management.\nClassify the severity of this incident as High, Medium, or Low:\nIncident: {message}"
    return call_bedrock(prompt)

def summarization_tool(message: str) -> str:
    prompt = f"You are an AI assistant for incident management.\nSummarize this incident in one concise sentence for responders:\nIncident: {message}"
    return call_bedrock(prompt)

def citizen_guidance_tool(message: str) -> str:
    prompt = f"You are an AI assistant for incident management.\nProvide safety guidance for citizens regarding this incident:\nIncident: {message}"
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
