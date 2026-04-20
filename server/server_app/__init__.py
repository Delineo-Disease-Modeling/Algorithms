def create_app(*args, **kwargs):
    from .app_factory import create_app as _create_app
    return _create_app(*args, **kwargs)
