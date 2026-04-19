import logging

from flask import Flask
from flask_cors import CORS

from .analysis_service import PreviewClusteringService
from .constants import CORS_ORIGINS, FULLSTACK_URL
from .fullstack_client import FullstackClient
from .generation_service import ConvenienceZoneGenerationService
from .jobs import ProgressStore
from .routes import register_routes


def create_app():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    CORS(
        app,
        origins=CORS_ORIGINS,
        methods=['GET', 'HEAD', 'PUT', 'PATCH', 'POST', 'DELETE'],
        allow_headers=['Content-Type', 'Authorization'],
        expose_headers=['Set-Cookie'],
        supports_credentials=True,
    )

    generation_store = ProgressStore()
    clustering_store = ProgressStore(with_results=True, with_counter=True)
    fullstack_client = FullstackClient(FULLSTACK_URL)
    generation_service = ConvenienceZoneGenerationService(fullstack_client, generation_store)
    analysis_service = PreviewClusteringService(clustering_store)

    app.config['generation_store'] = generation_store
    app.config['clustering_store'] = clustering_store
    app.config['fullstack_client'] = fullstack_client
    app.config['generation_service'] = generation_service
    app.config['analysis_service'] = analysis_service

    register_routes(
        app,
        generation_service=generation_service,
        analysis_service=analysis_service,
        generation_store=generation_store,
        clustering_store=clustering_store,
    )
    return app
