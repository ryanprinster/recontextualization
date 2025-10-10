"""Custom API error classes for better error handling."""

class APIError(Exception):
    """Base class for API-related errors."""
    pass


class ModelNotFoundError(APIError):
    """Raised when a model is not found or accessible with the current API key."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is hit."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass