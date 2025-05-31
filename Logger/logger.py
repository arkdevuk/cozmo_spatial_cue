import os


class LogLevel:
    LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    LEVEL_MAP = {name: idx for idx, name in enumerate(LEVELS)}

    @classmethod
    def is_valid(cls, level):
        return level.upper() in cls.LEVEL_MAP

    @classmethod
    def compare(cls, level, current_level):
        return cls.LEVEL_MAP.get(level.upper(), 99) >= cls.LEVEL_MAP.get(current_level.upper(), 0)


class Logger:

    @staticmethod
    def _desired_verbosity():
        return os.getenv("LOGS_VERBOSITY", "INFO").upper()

    @classmethod
    def _enabled(cls):
        return cls._desired_verbosity() != "FALSE"

    @classmethod
    def log(cls, message, level="INFO"):
        if not cls._enabled():
            return
        if not LogLevel.is_valid(level):
            raise ValueError(f"Invalid log level: {level}")
        if LogLevel.compare(level, cls._desired_verbosity()):
            # current date YYYY-MM-DD HH:MM:SS
            current_date_string = os.popen("date +'%Y-%m-%d %H:%M:%S'").read().strip()
            print(f"[{current_date_string}][{level.upper()}] : {message}")

    @classmethod
    def debug(cls, message):
        cls.log(message, "DEBUG")

    @classmethod
    def info(cls, message):
        cls.log(message, "INFO")

    @classmethod
    def warning(cls, message):
        cls.log(message, "WARNING")

    @classmethod
    def error(cls, message):
        cls.log(message, "ERROR")

    @classmethod
    def critical(cls, message):
        cls.log(message, "CRITICAL")