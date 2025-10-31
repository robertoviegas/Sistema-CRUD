from flask import Flask

from .config import Settings

settings = Settings()

_DEF_JSON_CFG = {
    "JSON_SORT_KEYS": False,
}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(_DEF_JSON_CFG)

    from .routes import register_routes

    register_routes(app)

    return app

