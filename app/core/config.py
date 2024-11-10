from functools import lru_cache
import logging
from typing import Dict, Type
import os
from app.core.settings.app import AppSettings
from app.core.settings.base import AppEnvTypes, BaseAppSettings
from app.core.settings.development import DevAppSettings
from app.core.settings.production import ProdAppSettings
from app.core.settings.test import TestAppSettings


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


environments: Dict[AppEnvTypes, Type[AppSettings]] = {
    AppEnvTypes.dev: DevAppSettings,
    AppEnvTypes.prod: ProdAppSettings,
    AppEnvTypes.test: TestAppSettings,
}



@lru_cache
def get_app_settings() -> AppSettings:
    app_env = BaseAppSettings().app_env
    config = environments[app_env]
    if not config:
        raise ValueError(f"Invalid environment: {app_env}")
    return config()
