import math
import re
from datetime import datetime

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

    def job_id_field(payload, key, allowed):
        if not isinstance(payload, dict) or key not in payload or set(payload) - allowed:
            return None, f'Provide {key} and only these optional fields: {sorted(allowed - {key})}.'
        value = payload[key]
        if type(value) is not int or value <= 0:
            return None, f'{key} must be a positive integer.'
        return value, None

    def seed_field(payload):
        seed = payload.get('seed')
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 4294967295):
            return None, 'seed must be an integer from 0 to 4294967295.'
        return seed, None

    @blueprint.post('/places')
    def places():
        payload = request.get_json(silent=True)
        census_id, error = job_id_field(payload, 'census_id', {'census_id'})
        if error:
            return jsonify(message=error), 400
        try:
            return jsonify(clustering_id=service.start_places_job(census_id))
        except ValueError as exc:
            return jsonify(message=str(exc)), 409

    @blueprint.post('/town')
    def town():
        payload = request.get_json(silent=True)
        places_id, error = job_id_field(payload, 'places_id', {'places_id', 'seed'})
        seed, seed_error = seed_field(payload) if not error else (None, None)
        if error or seed_error:
            return jsonify(message=error or seed_error), 400
        try:
            return jsonify(clustering_id=service.start_town_job(places_id, seed))
        except ValueError as exc:
            return jsonify(message=str(exc)), 409

    @blueprint.post('/patterns')
    def patterns():
        payload = request.get_json(silent=True)
        allowed = {'town_id', 'start_date', 'length', 'seed', 'publish'}
        town_id, error = job_id_field(payload, 'town_id', allowed)
        if error:
            return jsonify(message=error), 400
        seed, seed_error = seed_field(payload)
        if seed_error:
            return jsonify(message=seed_error), 400
        try:
            start_date = datetime.fromisoformat(str(payload.get('start_date', '')).replace('Z', '+00:00'))
        except ValueError:
            return jsonify(message='start_date must be an ISO 8601 date-time.'), 400
        length = payload.get('length', 168)
        if type(length) is not int or not 1 <= length <= 24 * 366:
            return jsonify(message='length must be a whole number of hours from 1 to 8784.'), 400
        publish = payload.get('publish')
        if publish is not None and (
                not isinstance(publish, dict) or not isinstance(publish.get('name'), str)
                or not publish['name'].strip()
                or set(publish) - {'name', 'description', 'user_id', 'guest_claim_token'}):
            return jsonify(message='publish needs a name, with optional description, '
                                   'user_id and guest_claim_token.'), 400
        try:
            return jsonify(clustering_id=service.start_patterns_job(town_id, start_date, length, seed, publish))
        except ValueError as exc:
            return jsonify(message=str(exc)), 409

    def artifact_response(job_id, key):
        data = service.get_artifact(job_id, key)
        if data is None:
            return jsonify(message='Data for this job is unavailable.'), 404
        return jsonify(data)

    @blueprint.get('/papdata/<int:job_id>')
    def papdata(job_id):
        return artifact_response(job_id, 'papdata')

    @blueprint.get('/patterns/<int:job_id>')
    def patterns_download(job_id):
        return artifact_response(job_id, 'patterns')

    return blueprint
