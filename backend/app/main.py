from fastapi import FastAPI, Request
from mangum import Mangum
import logging
import boto3
import json
import os
import traceback

# ===== Logging Setup =====
logger = logging.getLogger("SafeLinkMinimal")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = True
print("🔎 Lambda cold start - app initialized")
logger.info("✅ Logging configured successfully")

# ===== AWS Bedrock Client =====
REGION = os.getenv("AWS_REGION", "us-east-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.meta.llama3-1-70b-instruct-v1:0")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
logger.info("✅ Bedrock client initialized in %s", REGION)

# ===== FastAPI App =====
app = FastAPI(title="SafeLink Minimal Test")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("➡️ Incoming request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("⬅️ Response status: %s", response.status_code)
    return response

@app.get("/")
def health_check():
    logger.info("Health check called")
    return {"status": "ok"}

@app.post("/incident")
def handle_incident(payload: dict):
    logger.info("Received incident payload: %s", payload)
    return {"echo": payload}

# ===== Bedrock Test Endpoint =====
@app.post("/bedrock-test")
def bedrock_test(payload: dict):
    """
    Simple Bedrock test:
    Input: {"prompt": "Your text here"}
    """
    prompt = payload.get("prompt", "Say hello from Bedrock!")

    logger.info("➡️ Sending prompt to Bedrock: %s", prompt)

    try:
        body = json.dumps({"prompt": prompt})

        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        raw_output = json.loads(response["body"].read())
        logger.info("⬅️ Bedrock response: %s", raw_output)

        return {"bedrock_response": raw_output}

    except Exception as e:
        logger.error("❌ Bedrock call failed: %s", e)
        logger.debug(traceback.format_exc())
        return {"error": str(e)}

# ===== Lambda Handler =====
handler = Mangum(app)
logger.info("Mangum handler initialized")
