"""Version information for NormaML."""

__version__ = "1.0.0"
__version_info__ = tuple(int(i) for i in __version__.split('.'))

# Compatibility information
MIN_PYTHON_VERSION = (3, 9)
MAX_PYTHON_VERSION = (3, 12)

# Dependencies versions
MIN_POLARS_VERSION = "0.19.0"
MIN_PYARROW_VERSION = "10.0.0"
MIN_MLFLOW_VERSION = "2.0.0"

def check_python_version():
    """Check if current Python version is supported."""
    import sys
    current_version = sys.version_info[:2]
    
    if current_version < MIN_PYTHON_VERSION:
        raise RuntimeError(
            f"NormaML requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} "
            f"or later, but you are using Python {current_version[0]}.{current_version[1]}"
        )
    
    if current_version > MAX_PYTHON_VERSION:
        import warnings
        warnings.warn(
            f"NormaML has not been tested with Python {current_version[0]}.{current_version[1]}. "
            f"Maximum tested version is {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}",
            UserWarning
        )