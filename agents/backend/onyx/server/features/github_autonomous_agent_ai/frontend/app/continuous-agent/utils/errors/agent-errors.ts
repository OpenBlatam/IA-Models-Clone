/**
 * Custom error types for Continuous Agent feature
 * 
 * Provides structured error handling with consistent error types
 */

/**
 * Base error class for all agent-related errors
 */
export abstract class AgentError extends Error {
  readonly code: string;
  readonly statusCode?: number;

  constructor(
    message: string,
    code: string,
    statusCode?: number,
    options?: ErrorOptions
  ) {
    super(message, options);
    this.name = this.constructor.name;
    this.code = code;
    this.statusCode = statusCode;
    Error.captureStackTrace?.(this, this.constructor);
  }
}

/**
 * Error thrown when agent validation fails
 */
export class AgentValidationError extends AgentError {
  readonly field?: string;

  constructor(
    message: string,
    field?: string,
    options?: ErrorOptions
  ) {
    super(message, "VALIDATION_ERROR", 400, options);
    this.field = field;
  }
}

/**
 * Error thrown when agent is not found
 */
export class AgentNotFoundError extends AgentError {
  readonly agentId: string;

  constructor(agentId: string, options?: ErrorOptions) {
    super(
      `Agente con ID ${agentId} no encontrado`,
      "AGENT_NOT_FOUND",
      404,
      options
    );
    this.agentId = agentId;
  }
}

/**
 * Error thrown when agent operation fails due to insufficient credits
 */
export class InsufficientCreditsError extends AgentError {
  readonly required: number;
  readonly available: number;

  constructor(required: number, available: number, options?: ErrorOptions) {
    super(
      `Créditos insuficientes. Requeridos: ${required}, Disponibles: ${available}`,
      "INSUFFICIENT_CREDITS",
      402,
      options
    );
    this.required = required;
    this.available = available;
  }
}

/**
 * Error thrown when agent operation fails due to network issues
 */
export class AgentNetworkError extends AgentError {
  constructor(message: string = "Error de red al comunicarse con el servidor", options?: ErrorOptions) {
    super(message, "NETWORK_ERROR", 0, options);
  }
}

/**
 * Error thrown when agent operation times out
 */
export class AgentTimeoutError extends AgentError {
  readonly timeoutMs: number;

  constructor(timeoutMs: number, options?: ErrorOptions) {
    super(
      `La operación excedió el tiempo límite de ${timeoutMs}ms`,
      "TIMEOUT_ERROR",
      408,
      options
    );
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Error thrown when agent operation fails due to server error
 */
export class AgentServerError extends AgentError {
  constructor(message: string = "Error interno del servidor", options?: ErrorOptions) {
    super(message, "SERVER_ERROR", 500, options);
  }
}

/**
 * Type guard to check if error is an AgentError
 */
export function isAgentError(error: unknown): error is AgentError {
  return error instanceof AgentError;
}

/**
 * Converts unknown error to AgentError
 */
export function toAgentError(error: unknown): AgentError {
  if (isAgentError(error)) {
    return error;
  }

  if (error instanceof Error) {
    // Check for network errors
    if (
      error.message.includes("fetch") ||
      error.message.includes("network") ||
      error.message.includes("NetworkError") ||
      error.message.includes("Failed to fetch")
    ) {
      return new AgentNetworkError(error.message);
    }

    // Check for timeout errors
    if (
      error.message.includes("timeout") ||
      error.message.includes("Timeout") ||
      error.message.includes("timed out")
    ) {
      return new AgentTimeoutError(0, { cause: error });
    }

    // Default to server error
    return new AgentServerError(error.message, { cause: error });
  }

  // Unknown error type
  return new AgentServerError(
    typeof error === "string" ? error : "Error desconocido",
    { cause: error }
  );
}




