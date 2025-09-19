from fastapi import FastAPI, Request
from mangum import Mangum
import logging

# ===== Logging Setup =====
logger = logging.getLogger("SafeLinkMinimal")
logger.setLevel(logging.INFO)

# Attach a StreamHandler so logs always go to stdout (CloudWatch captures this)
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

# ===== Lambda Handler =====
handler = Mangum(app)
logger.info("Mangum handler initialized")
