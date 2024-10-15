from fastapi import APIRouter

from app.api.v2.routes import authentication, comments, profiles, tags, users
from app.api.v2.routes.articles import api as articles
from app.api.v2.routes import ml as ml_router
router = APIRouter()
router.include_router(authentication.router, tags=["v2/authentication"], prefix="/users")
router.include_router(users.router, tags=["v2/users"], prefix="/user")
router.include_router(profiles.router, tags=["v2/profiles"], prefix="/profiles")
# router.include_router(articles.router, tags=["v2/articles"])
# router.include_router(
#     comments.router,
#     tags=["v2/comments"],
#     prefix="/articles/{slug}/comments",
# )
# router.include_router(tags.router, tags=["v2/tags"], prefix="/tags")
router.include_router(ml_router.router, tags=["v2/ml"], prefix="/ml")