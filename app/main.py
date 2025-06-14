from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from pathlib import Path
from app.core.model_loader import model_loader
from app.config import settings
from app.api.routes import analyze,main, url, qr, payment, investment
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_all()
    yield
    model_loader.models.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates with absolute paths
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend/static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "frontend/templates"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⚡ Preloading ML models...")
    model_loader.load_all()  # Load all models during startup
    print("✅ Models preloaded")
    yield


# Include all routers
app.include_router(analyze.router)
app.include_router(main.router)
app.include_router(url.router)
app.include_router(qr.router)
app.include_router(payment.router)
app.include_router(investment.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
