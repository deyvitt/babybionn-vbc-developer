"""
FastAPI Application Setup
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .endpoints import set_global_orchestrator
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from ..utils.orchestrator_config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)

def create_app(orchestrator):
    """Create and configure FastAPI application"""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager"""
        # Startup
        logger.info("🚀 BabyBIONN API Starting...")
        
        # Make orchestrator available to endpoints
        app.state.orchestrator = orchestrator
        # Also store in global reference for endpoints
        set_global_orchestrator(orchestrator)
        
        yield
        
        # Shutdown
        logger.info("🛑 BabyBIONN API Shutting down...")
    
    app = FastAPI(
        title="BabyBIONN API",
        description="Enhanced BabyBIONN with Real Learning and Dynamic VNI Networking",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Import and include routers
    from .endpoints import router as api_router
    app.include_router(api_router)
    
    # Chat interface routes
    @app.get("/")
    async def serve_root():
        """Serve the chat interface at root"""
        from fastapi.responses import FileResponse
        import os
        # The HTML file is at /app/bionn_demo_chatbot.html (mounted via Docker volume)
        html_path = "/app/bionn_demo_chatbot.html"
        if os.path.exists(html_path):
            return FileResponse(html_path)
        return {"message": "BabyBIONN API", "docs": "/docs", "chat": "/chat"}

    @app.get("/chat")
    async def serve_chat():
        """Serve chat interface"""
        return await serve_root()

    return app 
