import math
import re

from flask import Blueprint, jsonify, request

from .lodes import COMMUTING_COVERAGE_TARGET


def create_blueprint(service):
    blueprint = Blueprint('v2_regions', __name__, url_prefix='/v2')

    @blueprint.post('/region')
    def region():
        payload = request.get_json(silent=True)
        if (not isinstance(payload, dict) or 'zip_code' not in payload
                or set(payload) - {'zip_code', 'coverage_percent'}):
            return jsonify(message='Provide zip_code and an optional coverage_percent.'), 400
        zip_code = payload['zip_code']
        if not isinstance(zip_code, str) or not re.fullmatch(r'[0-9]{5}', zip_code):
            return jsonify(message='zip_code must be exactly five digits.'), 400
        coverage_percent = payload.get('coverage_percent', COMMUTING_COVERAGE_TARGET * 100)
        if (isinstance(coverage_percent, bool) or not isinstance(coverage_percent, (int, float))
                or not 0 <= coverage_percent <= 100 or not math.isfinite(coverage_percent)):
            return jsonify(message='coverage_percent must be a number from 0 to 100.'), 400
        return jsonify(clustering_id=service.start_region_job(zip_code, coverage_percent / 100))

    @blueprint.post('/population')
    def population():
        payload = request.get_json(silent=True)
        if (not isinstance(payload, dict) or 'preview_id' not in payload
                or set(payload) - {'preview_id', 'seed'}):
            return jsonify(message='Provide preview_id and an optional integer seed.'), 400
        preview_id, seed = payload['preview_id'], payload.get('seed', 42)
        if type(preview_id) is not int or preview_id <= 0:
            return jsonify(message='preview_id must be a positive integer.'), 400
        if type(seed) is not int or not 0 <= seed <= 4294967295:
            return jsonify(message='seed must be an integer from 0 to 4294967295.'), 400
        try:
            job_id = service.start_population_job(preview_id, seed)
        except ValueError as exc:
            return jsonify(message=str(exc)), 409
        return jsonify(clustering_id=job_id)

    return blueprint
