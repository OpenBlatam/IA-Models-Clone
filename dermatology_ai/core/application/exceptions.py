class UseCaseError(Exception):
    pass


class ValidationError(UseCaseError):
    pass


class NotFoundError(UseCaseError):
    pass


class ProcessingError(UseCaseError):
    pass















