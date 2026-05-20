import {
  ApiError,
  NetworkError,
  ValidationError,
  isApiError,
  isNetworkError,
  isValidationError,
  getErrorMessage,
} from '@/lib/errors';

describe('Error Classes', () => {
  describe('ApiError', () => {
    it('should create ApiError with message', () => {
      const error = new ApiError('Test error');
      expect(error.message).toBe('Test error');
      expect(error.name).toBe('ApiError');
      expect(error.statusCode).toBeUndefined();
    });

    it('should create ApiError with status code', () => {
      const error = new ApiError('Not found', 404);
      expect(error.statusCode).toBe(404);
    });

    it('should create ApiError with response data', () => {
      const responseData = { detail: 'Resource not found' };
      const error = new ApiError('Not found', 404, responseData);
      expect(error.response).toEqual(responseData);
    });
  });

  describe('NetworkError', () => {
    it('should create NetworkError with default message', () => {
      const error = new NetworkError();
      expect(error.message).toBe('Network error occurred');
      expect(error.name).toBe('NetworkError');
    });

    it('should create NetworkError with custom message', () => {
      const error = new NetworkError('Custom network error');
      expect(error.message).toBe('Custom network error');
    });
  });

  describe('ValidationError', () => {
    it('should create ValidationError with message', () => {
      const error = new ValidationError('Validation failed');
      expect(error.message).toBe('Validation failed');
      expect(error.name).toBe('ValidationError');
    });

    it('should create ValidationError with field', () => {
      const error = new ValidationError('Invalid email', 'email');
      expect(error.field).toBe('email');
    });

    it('should create ValidationError with errors object', () => {
      const errors = { email: ['Invalid format'], password: ['Too short'] };
      const error = new ValidationError('Validation failed', 'form', errors);
      expect(error.errors).toEqual(errors);
    });
  });
});

describe('Type Guards', () => {
  describe('isApiError', () => {
    it('should return true for ApiError instances', () => {
      const error = new ApiError('Test');
      expect(isApiError(error)).toBe(true);
    });

    it('should return false for other error types', () => {
      expect(isApiError(new Error('Test'))).toBe(false);
      expect(isApiError(new NetworkError())).toBe(false);
      expect(isApiError('string')).toBe(false);
    });
  });

  describe('isNetworkError', () => {
    it('should return true for NetworkError instances', () => {
      const error = new NetworkError();
      expect(isNetworkError(error)).toBe(true);
    });

    it('should return false for other error types', () => {
      expect(isNetworkError(new Error('Test'))).toBe(false);
      expect(isNetworkError(new ApiError('Test'))).toBe(false);
    });
  });

  describe('isValidationError', () => {
    it('should return true for ValidationError instances', () => {
      const error = new ValidationError('Test');
      expect(isValidationError(error)).toBe(true);
    });

    it('should return false for other error types', () => {
      expect(isValidationError(new Error('Test'))).toBe(false);
      expect(isValidationError(new ApiError('Test'))).toBe(false);
    });
  });
});

describe('getErrorMessage', () => {
  it('should return message from ApiError', () => {
    const error = new ApiError('API error message');
    expect(getErrorMessage(error)).toBe('API error message');
  });

  it('should return default message for NetworkError', () => {
    const error = new NetworkError();
    expect(getErrorMessage(error)).toBe(
      'Network connection failed. Please check your internet connection.'
    );
  });

  it('should return message from ValidationError', () => {
    const error = new ValidationError('Validation failed');
    expect(getErrorMessage(error)).toBe('Validation failed');
  });

  it('should return message from standard Error', () => {
    const error = new Error('Standard error');
    expect(getErrorMessage(error)).toBe('Standard error');
  });

  it('should return default message for unknown errors', () => {
    expect(getErrorMessage('string')).toBe('An unexpected error occurred');
    expect(getErrorMessage(null)).toBe('An unexpected error occurred');
    expect(getErrorMessage(undefined)).toBe('An unexpected error occurred');
    expect(getErrorMessage(123)).toBe('An unexpected error occurred');
  });
});

