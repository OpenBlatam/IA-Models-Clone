/**
 * Domain-specific errors
 * Pattern: Custom error types for better error handling
 */

export class AgentError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode?: number
  ) {
    super(message);
    this.name = "AgentError";
    Object.setPrototypeOf(this, AgentError.prototype);
  }
}

export class AgentNotFoundError extends AgentError {
  constructor(agentId: string) {
    super(`Agent with id ${agentId} not found`, "AGENT_NOT_FOUND", 404);
    this.name = "AgentNotFoundError";
  }
}

export class AgentValidationError extends AgentError {
  constructor(message: string, public readonly field?: string) {
    super(message, "AGENT_VALIDATION_ERROR", 400);
    this.name = "AgentValidationError";
  }
}

export class AgentOperationError extends AgentError {
  constructor(message: string, public readonly operation: string) {
    super(message, "AGENT_OPERATION_ERROR", 500);
    this.name = "AgentOperationError";
  }
}






