import logging
import datetime
from logging import getLogger, StreamHandler, Formatter, FileHandler
from utils import file_operator as f_op

logger_ = None
writer_ = None


def create_logger(logger_name, log_root_path, save_key, level=logging.INFO):
    # logger object
    logger = getLogger(f"{logger_name}")

    # set debug log level, level is controlled in handler
    logger.setLevel(logging.DEBUG)

    # create log handler
    stream_handler = StreamHandler()
    f_op.create_folder(log_root_path)
    prefix = get_cur_datetime()
    file_handler = FileHandler(f'{log_root_path}/{save_key}_{prefix}.log', 'a')

    # set log level
    stream_handler.setLevel(level)
    file_handler.setLevel(level)

    # set format
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)
    file_handler.setFormatter(handler_format)

    # set handler to logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_log_level_from_name(log_level_name: str) -> int:
    if log_level_name == "DEBUG":
        return logging.DEBUG
    elif log_level_name == "INFO":
        return logging.INFO
    else:
        assert False, f"Log level is either INFO or DEBUG"


def get_cur_datetime():
    dt_now = datetime.datetime.now()
    simple_form = dt_now.strftime("%Y%m%d_%H%M%S")
    return simple_form
