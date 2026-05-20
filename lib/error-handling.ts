export class AppError extends Error {
  public readonly statusCode: number;
  public readonly isOperational: boolean;
  public readonly timestamp: string;
  public readonly context?: Record<string, any>;

  constructor(
    message: string,
    statusCode: number = 500,
    isOperational: boolean = true,
    context?: Record<string, any>
  ) {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    this.timestamp = new Date().toISOString();
    this.context = context;

    Error.captureStackTrace(this, this.constructor);
  }
}

export class ValidationError extends AppError {
  constructor(message: string, context?: Record<string, any>) {
    super(message, 400, true, context);
  }
}

export class NotFoundError extends AppError {
  constructor(resource: string, id?: string) {
    const message = id ? `${resource} with id ${id} not found` : `${resource} not found`;
    super(message, 404, true, { resource, id });
  }
}

export class RateLimitError extends AppError {
  constructor(limit: number, windowMs: number) {
    super(`Rate limit exceeded: ${limit} requests per ${windowMs}ms`, 429, true, { limit, windowMs });
  }
}

export class ExternalServiceError extends AppError {
  constructor(service: string, originalError?: Error) {
    super(`External service error: ${service}`, 502, true, { 
      service, 
      originalMessage: originalError?.message 
    });
  }
}

export interface RetryOptions {
  maxAttempts: number;
  baseDelay: number;
  maxDelay: number;
  exponentialBase: number;
  jitter: boolean;
}

export const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxAttempts: 3,
  baseDelay: 1000,
  maxDelay: 10000,
  exponentialBase: 2,
  jitter: true,
};

export async function withRetry<T>(
  operation: () => Promise<T>,
  options: Partial<RetryOptions> = {}
): Promise<T> {
  const config = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastError: Error;

  for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error as Error;
      
      if (attempt === config.maxAttempts) {
        throw new AppError(
          `Operation failed after ${config.maxAttempts} attempts: ${lastError.message}`,
          500,
          true,
          { attempts: attempt, originalError: lastError.message }
        );
      }

      const delay = Math.min(
        config.baseDelay * Math.pow(config.exponentialBase, attempt - 1),
        config.maxDelay
      );

      const jitteredDelay = config.jitter 
        ? delay + Math.random() * delay * 0.1 
        : delay;

      console.warn(`Attempt ${attempt} failed, retrying in ${jitteredDelay}ms:`, lastError.message);
      await new Promise(resolve => setTimeout(resolve, jitteredDelay));
    }
  }

  throw lastError!;
}

export function isRetryableError(error: Error): boolean {
  if (error instanceof AppError) {
    return error.statusCode >= 500 || error.statusCode === 429;
  }
  
  const retryableMessages = [
    'ECONNRESET',
    'ENOTFOUND',
    'ECONNREFUSED',
    'ETIMEDOUT',
    'timeout',
    'network',
    'fetch failed'
  ];
  
  return retryableMessages.some(msg => 
    error.message.toLowerCase().includes(msg.toLowerCase())
  );
}

export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage?: string
): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => {
      reject(new AppError(
        timeoutMessage || `Operation timed out after ${timeoutMs}ms`,
        408,
        true,
        { timeoutMs }
      ));
    }, timeoutMs);
  });

  return Promise.race([promise, timeoutPromise]);
}

export function handleAsyncError(error: unknown): AppError {
  if (error instanceof AppError) {
    return error;
  }

  if (error instanceof Error) {
    return new AppError(error.message, 500, true, { originalError: error.name });
  }

  return new AppError('Unknown error occurred', 500, false, { error: String(error) });
}

export function logError(error: AppError, context?: Record<string, any>): void {
  const logData = {
    message: error.message,
    statusCode: error.statusCode,
    timestamp: error.timestamp,
    stack: error.stack,
    context: { ...error.context, ...context },
    isOperational: error.isOperational
  };

  if (error.statusCode >= 500) {
    console.error('Server Error:', logData);
  } else {
    console.warn('Client Error:', logData);
  }
}
