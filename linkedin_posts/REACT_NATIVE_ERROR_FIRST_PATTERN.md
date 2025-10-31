# React Native Error-First Pattern

## Core Principles

### 1. Error-First Function Structure
```typescript
// ❌ Bad: Error handling scattered
function processUserData(user: User) {
  const processed = transformData(user);
  if (!processed) {
    console.error('Failed to transform data');
    return null;
  }
  
  const result = saveData(processed);
  if (!result) {
    console.error('Failed to save data');
    return null;
  }
  
  return result;
}

// ✅ Good: Error handling at the beginning
function processUserData(user: User): Result<ProcessedData> {
  // Validate input first
  if (!user || !user.id) {
    return Result.failure(new ValidationError('Invalid user data'));
  }
  
  // Check permissions early
  if (!hasPermission(user.id, 'write')) {
    return Result.failure(new PermissionError('Insufficient permissions'));
  }
  
  // Validate data integrity
  if (!isDataValid(user)) {
    return Result.failure(new DataIntegrityError('Invalid data structure'));
  }
  
  // Happy path - all errors handled
  const processed = transformData(user);
  const result = saveData(processed);
  
  return Result.success(result);
}
```

### 2. Custom Error Types
```typescript
// Error hierarchy
class AppError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500
  ) {
    super(message);
    this.name = 'AppError';
  }
}

class ValidationError extends AppError {
  constructor(message: string) {
    super(message, 'VALIDATION_ERROR', 400);
    this.name = 'ValidationError';
  }
}

class NetworkError extends AppError {
  constructor(message: string) {
    super(message, 'NETWORK_ERROR', 503);
    this.name = 'NetworkError';
  }
}

class PermissionError extends AppError {
  constructor(message: string) {
    super(message, 'PERMISSION_ERROR', 403);
    this.name = 'PermissionError';
  }
}
```

### 3. Result Pattern Implementation
```typescript
// Result type for error handling
class Result<T> {
  private constructor(
    private isSuccess: boolean,
    private value?: T,
    private error?: AppError
  ) {}

  static success<T>(value: T): Result<T> {
    return new Result<T>(true, value);
  }

  static failure<T>(error: AppError): Result<T> {
    return new Result<T>(false, undefined, error);
  }

  isOk(): boolean {
    return this.isSuccess;
  }

  getValue(): T {
    if (!this.isSuccess) {
      throw new Error('Cannot get value from failed result');
    }
    return this.value!;
  }

  getError(): AppError {
    if (this.isSuccess) {
      throw new Error('Cannot get error from successful result');
    }
    return this.error!;
  }

  map<U>(fn: (value: T) => U): Result<U> {
    if (!this.isSuccess) {
      return Result.failure(this.error!);
    }
    return Result.success(fn(this.value!));
  }

  flatMap<U>(fn: (value: T) => Result<U>): Result<U> {
    if (!this.isSuccess) {
      return Result.failure(this.error!);
    }
    return fn(this.value!);
  }
}
```

### 4. API Error Handling
```typescript
// API service with error-first approach
class ApiService {
  async fetchUserData(userId: string): Promise<Result<UserData>> {
    // Validate input first
    if (!userId || typeof userId !== 'string') {
      return Result.failure(new ValidationError('Invalid user ID'));
    }

    // Check network connectivity
    if (!await this.isNetworkAvailable()) {
      return Result.failure(new NetworkError('No network connection'));
    }

    try {
      const response = await fetch(`/api/users/${userId}`);
      
      // Handle HTTP errors first
      if (!response.ok) {
        if (response.status === 404) {
          return Result.failure(new ValidationError('User not found'));
        }
        if (response.status === 403) {
          return Result.failure(new PermissionError('Access denied'));
        }
        return Result.failure(new NetworkError(`HTTP ${response.status}`));
      }

      const data = await response.json();
      
      // Validate response data
      if (!this.isValidUserData(data)) {
        return Result.failure(new ValidationError('Invalid response format'));
      }

      return Result.success(data);
    } catch (error) {
      return Result.failure(new NetworkError(error.message));
    }
  }

  private async isNetworkAvailable(): Promise<boolean> {
    try {
      const response = await fetch('/api/health', { timeout: 5000 });
      return response.ok;
    } catch {
      return false;
    }
  }

  private isValidUserData(data: any): data is UserData {
    return data && typeof data.id === 'string' && typeof data.name === 'string';
  }
}
```

### 5. React Hook with Error-First Pattern
```typescript
// Custom hook with error handling
function useUserData(userId: string) {
  const [state, setState] = useState<{
    data: UserData | null;
    error: AppError | null;
    loading: boolean;
  }>({
    data: null,
    error: null,
    loading: true
  });

  useEffect(() => {
    async function loadUserData() {
      // Reset state and start loading
      setState({ data: null, error: null, loading: true });

      // Validate input first
      if (!userId) {
        setState({
          data: null,
          error: new ValidationError('User ID is required'),
          loading: false
        });
        return;
      }

      const result = await apiService.fetchUserData(userId);

      if (result.isOk()) {
        setState({
          data: result.getValue(),
          error: null,
          loading: false
        });
      } else {
        setState({
          data: null,
          error: result.getError(),
          loading: false
        });
      }
    }

    loadUserData();
  }, [userId]);

  return state;
}
```

### 6. Component Error Boundaries
```typescript
// Error boundary component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: AppError | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error to monitoring service
    Sentry.captureException(error, { extra: errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return <ErrorFallback error={this.state.error} />;
    }

    return this.props.children;
  }
}

// Error fallback component
function ErrorFallback({ error }: { error: AppError | null }) {
  return (
    <View style={styles.errorContainer}>
      <Text style={styles.errorTitle}>Something went wrong</Text>
      <Text style={styles.errorMessage}>
        {error?.message || 'An unexpected error occurred'}
      </Text>
      <TouchableOpacity
        style={styles.retryButton}
        onPress={() => window.location.reload()}
      >
        <Text style={styles.retryText}>Try Again</Text>
      </TouchableOpacity>
    </View>
  );
}
```

### 7. Form Validation with Error-First
```typescript
// Form validation hook
function useFormValidation<T extends Record<string, any>>(
  initialData: T,
  validationSchema: ValidationSchema<T>
) {
  const [data, setData] = useState<T>(initialData);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});

  const validate = useCallback((fieldData: T): Result<T> => {
    // Validate all fields first
    const validationErrors: Partial<Record<keyof T, string>> = {};

    for (const [field, rules] of Object.entries(validationSchema)) {
      const value = fieldData[field as keyof T];
      
      // Check required fields first
      if (rules.required && !value) {
        validationErrors[field as keyof T] = `${field} is required`;
        continue;
      }

      // Validate field type
      if (value && rules.type && typeof value !== rules.type) {
        validationErrors[field as keyof T] = `${field} must be ${rules.type}`;
        continue;
      }

      // Validate field length
      if (value && rules.minLength && value.length < rules.minLength) {
        validationErrors[field as keyof T] = `${field} must be at least ${rules.minLength} characters`;
        continue;
      }

      // Custom validation
      if (value && rules.validate) {
        const customError = rules.validate(value);
        if (customError) {
          validationErrors[field as keyof T] = customError;
        }
      }
    }

    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return Result.failure(new ValidationError('Form validation failed'));
    }

    setErrors({});
    return Result.success(fieldData);
  }, [validationSchema]);

  const updateField = useCallback((field: keyof T, value: any) => {
    const newData = { ...data, [field]: value };
    setData(newData);
    
    // Clear field error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  }, [data, errors]);

  const submit = useCallback(async (): Promise<Result<T>> => {
    const validationResult = validate(data);
    
    if (!validationResult.isOk()) {
      return validationResult;
    }

    // Proceed with submission
    try {
      await submitData(validationResult.getValue());
      return Result.success(validationResult.getValue());
    } catch (error) {
      return Result.failure(new NetworkError('Failed to submit form'));
    }
  }, [data, validate]);

  return {
    data,
    errors,
    updateField,
    validate,
    submit,
    isValid: Object.keys(errors).length === 0
  };
}
```

### 8. Async Operations with Error-First
```typescript
// Async operation wrapper
class AsyncOperation<T> {
  private operation: () => Promise<T>;

  constructor(operation: () => Promise<T>) {
    this.operation = operation;
  }

  async execute(): Promise<Result<T>> {
    try {
      // Validate operation exists
      if (!this.operation) {
        return Result.failure(new ValidationError('Operation not defined'));
      }

      const result = await this.operation();
      
      // Validate result
      if (result === null || result === undefined) {
        return Result.failure(new ValidationError('Operation returned null'));
      }

      return Result.success(result);
    } catch (error) {
      if (error instanceof AppError) {
        return Result.failure(error);
      }
      
      return Result.failure(new NetworkError(error.message));
    }
  }

  withRetry(maxRetries: number = 3): AsyncOperation<T> {
    return new AsyncOperation(async () => {
      let lastError: AppError;

      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        const result = await this.execute();
        
        if (result.isOk()) {
          return result.getValue();
        }

        lastError = result.getError();
        
        // Don't retry on validation errors
        if (lastError instanceof ValidationError) {
          break;
        }

        // Wait before retry
        if (attempt < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
      }

      throw lastError!;
    });
  }
}

// Usage example
const userOperation = new AsyncOperation(async () => {
  const response = await fetch('/api/user');
  if (!response.ok) {
    throw new NetworkError(`HTTP ${response.status}`);
  }
  return response.json();
});

const result = await userOperation.withRetry(3).execute();
if (result.isOk()) {
  console.log('User data:', result.getValue());
} else {
  console.error('Error:', result.getError().message);
}
```

### 9. Error Logging and Monitoring
```typescript
// Error logging service
class ErrorLogger {
  private static instance: ErrorLogger;
  private sentry: typeof Sentry;

  private constructor() {
    this.sentry = Sentry;
    this.setupSentry();
  }

  static getInstance(): ErrorLogger {
    if (!ErrorLogger.instance) {
      ErrorLogger.instance = new ErrorLogger();
    }
    return ErrorLogger.instance;
  }

  private setupSentry() {
    this.sentry.init({
      dsn: process.env.SENTRY_DSN,
      environment: process.env.NODE_ENV,
      beforeSend: (event) => {
        // Filter out validation errors in development
        if (process.env.NODE_ENV === 'development' && 
            event.exception?.values?.[0]?.type === 'ValidationError') {
          return null;
        }
        return event;
      }
    });
  }

  logError(error: AppError, context?: Record<string, any>) {
    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('Error:', error.message, context);
    }

    // Send to Sentry
    this.sentry.captureException(error, {
      extra: {
        code: error.code,
        statusCode: error.statusCode,
        ...context
      },
      tags: {
        errorType: error.constructor.name,
        environment: process.env.NODE_ENV
      }
    });
  }

  logWarning(message: string, context?: Record<string, any>) {
    this.sentry.captureMessage(message, {
      level: 'warning',
      extra: context
    });
  }
}

// Global error handler
const globalErrorHandler = (error: Error, isFatal?: boolean) => {
  const logger = ErrorLogger.getInstance();
  
  if (error instanceof AppError) {
    logger.logError(error, { isFatal });
  } else {
    logger.logError(new AppError(error.message, 'UNKNOWN_ERROR'), { isFatal });
  }
};

// Set up global error handling
if (__DEV__) {
  console.log('Setting up global error handler');
}
ErrorUtils.setGlobalHandler(globalErrorHandler);
```

### 10. Testing Error Scenarios
```typescript
// Test error handling
describe('Error-First Pattern Tests', () => {
  test('should handle validation errors first', async () => {
    const result = await processUserData(null);
    
    expect(result.isOk()).toBe(false);
    expect(result.getError()).toBeInstanceOf(ValidationError);
    expect(result.getError().message).toBe('Invalid user data');
  });

  test('should handle permission errors early', async () => {
    const user = { id: '123', name: 'Test' };
    mockPermissionCheck.mockReturnValue(false);
    
    const result = await processUserData(user);
    
    expect(result.isOk()).toBe(false);
    expect(result.getError()).toBeInstanceOf(PermissionError);
  });

  test('should return success when all validations pass', async () => {
    const user = { id: '123', name: 'Test' };
    mockPermissionCheck.mockReturnValue(true);
    mockTransformData.mockReturnValue({ processed: true });
    mockSaveData.mockReturnValue({ saved: true });
    
    const result = await processUserData(user);
    
    expect(result.isOk()).toBe(true);
    expect(result.getValue()).toEqual({ saved: true });
  });
});
```

## Best Practices Summary

1. **Validate inputs first** - Check all parameters and data before processing
2. **Handle errors early** - Return error results immediately when validation fails
3. **Use custom error types** - Create specific error classes for different scenarios
4. **Implement Result pattern** - Use Result<T> type for consistent error handling
5. **Log errors properly** - Use structured logging with context and monitoring
6. **Test error scenarios** - Write tests for all error paths
7. **Provide user feedback** - Show appropriate error messages to users
8. **Retry with caution** - Only retry network operations, not validation errors
9. **Use error boundaries** - Catch and handle React component errors
10. **Monitor in production** - Use services like Sentry for error tracking 