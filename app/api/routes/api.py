from fastapi import APIRouter

from app.api.v1.routes import api as api_v1
from app.api.v2.routes import api as api_v2

router = APIRouter()

router.include_router(api_v1.router, prefix="/v1" )
router.include_router(api_v2.router, prefix="/v2")