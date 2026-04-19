class ApiError(Exception):
    def __init__(self, message, status_code=400, extra=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.extra = extra or {}

    def to_payload(self):
        payload = {'message': self.message}
        payload.update(self.extra)
        return payload
