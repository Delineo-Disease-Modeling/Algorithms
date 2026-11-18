"""Town -> papdata, the people/homes/places dict the existing pipeline stores and the simulator reads.

Keys the legacy producer emits are kept (sex, age, home, home_cbg, is_worker,
work_poi, is_student, school_poi, ...) and the v2 agent attributes are added.
Fields the simulator treats as optional (area, catchment_fj) stay None because
OSM supplies no visitor-catchment or footprint data here.
"""
TOP_CATEGORIES = {
    'school': 'Schools', 'grocery': 'Grocery Stores', 'pharmacy': 'Pharmacies',
    'clinic': 'Health Care', 'workplace': 'Workplaces',
}
EXTERNAL_WORKPLACE = ('out_of_zone_work', 'Out of Zone Work')
EXTERNAL_SCHOOL = ('out_of_zone_school', 'Out of Zone School')


def _text(value):
    return None if value is None else str(value)


def _external_place(kind, label):
    return {'placekey': None, 'label': label, 'cbg': None, 'latitude': None, 'longitude': None,
            'top_category': 'External', 'street_address': None, 'postal_code': None,
            'naics_code': None, 'footprint': None, 'area': None, 'catchment_fj': None,
            'external_location_type': kind}


def to_papdata(town):
    places = {
        str(p.place_id): {
            'placekey': p.osm_id, 'label': p.name, 'cbg': p.cbg,
            'latitude': p.latitude, 'longitude': p.longitude,
            'top_category': TOP_CATEGORIES[p.function], 'street_address': p.street_address,
            'postal_code': p.postal_code, 'naics_code': None, 'footprint': None,
            'area': None, 'catchment_fj': None,
            'function': p.function, 'school_level': p.school_level,
        } for p in town.places
    }
    homes = {
        str(h.home_id): {
            'cbg': h.cbg, 'members': h.members, 'latitude': h.latitude, 'longitude': h.longitude,
            'county': h.county, 'household_type': h.household_type, 'has_children': h.has_children,
        } for h in town.homes
    }
    people = {
        str(p.person_id): {
            'sex': p.sex, 'age': p.age, 'home': str(p.home_id), 'home_cbg': p.home_cbg,
            'county': p.county, 'household_size': p.household_size, 'household_type': p.household_type,
            'employment_status': p.employment_status, 'role': p.role,
            'is_worker': p.role == 'worker', 'work_location_type': p.work_location_type,
            'work_poi': _text(p.work_id), 'work_cbg': p.work_cbg, 'work_assignment': p.work_assignment,
            'is_student': p.role == 'student', 'school_location_type': p.school_location_type,
            'school_poi': _text(p.school_id), 'school_level': p.school_level,
            'grocery_poi': _text(p.grocery_id), 'healthcare_poi': _text(p.healthcare_id),
        } for p in town.people
    }
    papdata = {'people': people, 'homes': homes, 'places': places}

    next_id = len(places) + 1
    if any(p['work_location_type'] == 'out_of_zone' for p in people.values()):
        places[str(next_id)] = _external_place(*EXTERNAL_WORKPLACE)
        papdata['external_workplace_id'] = str(next_id)
        next_id += 1
    if any(p['school_location_type'] == 'out_of_zone' for p in people.values()):
        places[str(next_id)] = _external_place(*EXTERNAL_SCHOOL)
        papdata['external_school_id'] = str(next_id)
    return papdata
