/**
 * Type definitions for Continuous Agent feature
 * 
 * All types are now inferred from Zod schemas for runtime type safety.
 * This ensures consistency between TypeScript types and runtime validation.
 */

// Re-export types from Zod schemas (single source of truth)
export type {
  TaskType,
  AgentConfig,
  AgentStats,
  ContinuousAgent,
  CreateAgentRequest,
  UpdateAgentRequest,
  AgentExecutionLog,
  AgentFormValues,
} from "../utils/validation/zod-schemas";

// Export shared component prop types
export * from "./shared";







