"""
Application Layer Exceptions

Exceptions specific to the application layer.
"""


class UseCaseException(Exception):
    """Base exception for use cases"""
    pass


class TrackNotFoundException(UseCaseException):
    """Raised when a track is not found"""
    pass


class InvalidTrackIDException(UseCaseException):
    """Raised when a track ID is invalid"""
    pass


class AnalysisException(UseCaseException):
    """Raised when analysis fails"""
    pass


class RecommendationException(UseCaseException):
    """Raised when recommendation generation fails"""
    pass


class CoachingException(UseCaseException):
    """Raised when coaching generation fails"""
    pass




