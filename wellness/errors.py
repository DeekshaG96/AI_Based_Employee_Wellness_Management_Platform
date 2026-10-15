"""Domain errors raised by the wellness analysis package."""


class WellnessError(ValueError):
    """Base class for expected, user-correctable input or setup errors."""


class InputValidationError(WellnessError):
    """Raised when supplied text or file input is invalid."""


class NLTKResourceError(WellnessError):
    """Raised when required NLTK data has not been installed."""
