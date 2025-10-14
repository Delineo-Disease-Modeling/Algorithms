gen_cz_schema = {
  'type': 'object',
  "properties": {
    'user_id': { 'type': 'string', 'format': 'uuid' },
    'name': { 'type': 'string' },
    'description': { 'type': 'string' },
    'cbg': { 'type': 'string', 'pattern': '[0-9]{12}'  },
    'min_pop': { 'type': 'integer' },
    'start_date': { 'type': 'string', 'format': 'date-time' },
    'length': { 'type': 'integer' }
  },
  'required': [ 'user_id', 'name', 'description', 'cbg', 'min_pop', 'start_date' ]
}
