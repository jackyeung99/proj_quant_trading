class QBTError(Exception):
    """Base exception for the backtesting system."""

class DataError(QBTError):
    pass

class StorageError(QBTError):
    pass

class InvalidRunSpec(QBTError):
    pass
