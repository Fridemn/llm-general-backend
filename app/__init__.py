from app.config.app_config import AppConfig
from app.utils.log import LogManager, LogBroker
from app.utils.database import get_redis


app_config = AppConfig()
log_broker = LogBroker()
logger = LogManager.GetLogger(log_name="app")
LogManager.set_queue_handler(logger, log_broker)
redis_client = get_redis()
