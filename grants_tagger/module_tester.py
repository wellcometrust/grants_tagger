import logging

logger = logging.getLogger(__name__)

development_dependencies = False
try:
    import pecos
    import wellcomeml

    development_dependencies = True
except ImportError:
    logger.warning("Development dependencies not installed.")
    pass


def test_development_dependencies():
    if development_dependencies:
        return True
    else:
        raise (
            "Development dependencies not installed. Run make virtualenv-dev to install them."
        )
