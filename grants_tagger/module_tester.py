
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