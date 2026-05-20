/**
 * Zod validation schemas for Continuous Agent feature
 * 
 * Provides runtime type validation with comprehensive error messages
 * and type inference for TypeScript
 */

import { z } from "zod";
import { TASK_TYPES, FORM_DEFAULTS, VALIDATION_LIMITS } from "../../constants";

/**
 * Task type schema with enum validation
 */
export const taskTypeSchema = z.enum([
  TASK_TYPES.CONTENT_GENERATION,
  TASK_TYPES.DATA_PROCESSING,
  TASK_TYPES.API_MONITORING,
  TASK_TYPES.AUTOMATED_RESEARCH,
  TASK_TYPES.CUSTOM,
]);

/**
 * Agent configuration schema
 */
export const agentConfigSchema = z.object({
  taskType: taskTypeSchema,
  frequency: z
    .number()
    .int()
    .positive()
    .min(FORM_DEFAULTS.MIN_FREQUENCY, {
      message: `La frecuencia mínima es ${FORM_DEFAULTS.MIN_FREQUENCY} segundos`,
    }),
  parameters: z.record(z.unknown()).default({}),
  goal: z
    .string()
    .max(10000, {
      message: "El objetivo no puede exceder 10000 caracteres",
    })
    .optional(),
});

/**
 * Agent stats schema
 */
export const agentStatsSchema = z.object({
  totalExecutions: z.number().int().nonnegative().default(0),
  successfulExecutions: z.number().int().nonnegative().default(0),
  failedExecutions: z.number().int().nonnegative().default(0),
  lastExecutionAt: z.string().nullable().default(null),
  nextExecutionAt: z.string().nullable().default(null),
  creditsUsed: z.number().nonnegative().default(0),
  averageExecutionTime: z.number().nonnegative().default(0),
});

/**
 * Continuous agent schema
 */
export const continuousAgentSchema = z.object({
  id: z.string().min(1, "El ID del agente es requerido"),
  name: z
    .string()
    .min(VALIDATION_LIMITS.MIN_NAME_LENGTH, {
      message: `El nombre debe tener al menos ${VALIDATION_LIMITS.MIN_NAME_LENGTH} caracteres`,
    })
    .max(VALIDATION_LIMITS.MAX_NAME_LENGTH, {
      message: `El nombre no puede exceder ${VALIDATION_LIMITS.MAX_NAME_LENGTH} caracteres`,
    })
    .trim(),
  description: z
    .string()
    .min(1, "La descripción es requerida")
    .max(VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH, {
      message: `La descripción no puede exceder ${VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH} caracteres`,
    })
    .trim(),
  isActive: z.boolean().default(false),
  config: agentConfigSchema,
  stats: agentStatsSchema,
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
  stripeCreditsRemaining: z.number().nullable().default(null),
});

/**
 * Create agent request schema
 */
export const createAgentRequestSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_LIMITS.MIN_NAME_LENGTH, {
      message: `El nombre debe tener al menos ${VALIDATION_LIMITS.MIN_NAME_LENGTH} caracteres`,
    })
    .max(VALIDATION_LIMITS.MAX_NAME_LENGTH, {
      message: `El nombre no puede exceder ${VALIDATION_LIMITS.MAX_NAME_LENGTH} caracteres`,
    })
    .trim(),
  description: z
    .string()
    .min(1, "La descripción es requerida")
    .max(VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH, {
      message: `La descripción no puede exceder ${VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH} caracteres`,
    })
    .trim(),
  config: agentConfigSchema,
});

/**
 * Update agent request schema (all fields optional)
 */
export const updateAgentRequestSchema = z
  .object({
    name: z
      .string()
      .min(VALIDATION_LIMITS.MIN_NAME_LENGTH, {
        message: `El nombre debe tener al menos ${VALIDATION_LIMITS.MIN_NAME_LENGTH} caracteres`,
      })
      .max(VALIDATION_LIMITS.MAX_NAME_LENGTH, {
        message: `El nombre no puede exceder ${VALIDATION_LIMITS.MAX_NAME_LENGTH} caracteres`,
      })
      .trim()
      .optional(),
    description: z
      .string()
      .min(1, "La descripción es requerida")
      .max(VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH, {
        message: `La descripción no puede exceder ${VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH} caracteres`,
      })
      .trim()
      .optional(),
    isActive: z.boolean().optional(),
    config: agentConfigSchema.optional(),
  })
  .refine(
    (data) => Object.keys(data).length > 0,
    {
      message: "Al menos un campo debe ser proporcionado para actualizar",
    }
  );

/**
 * Agent execution log schema
 */
export const agentExecutionLogSchema = z.object({
  id: z.string().min(1),
  agentId: z.string().min(1),
  status: z.enum(["success", "failed", "skipped"]),
  startedAt: z.string().datetime(),
  completedAt: z.string().datetime().nullable(),
  error: z.string().nullable(),
  creditsUsed: z.number().nonnegative(),
  executionTimeMs: z.number().nonnegative(),
});

/**
 * Type exports inferred from schemas
 */
export type TaskType = z.infer<typeof taskTypeSchema>;
export type AgentConfig = z.infer<typeof agentConfigSchema>;
export type AgentStats = z.infer<typeof agentStatsSchema>;
export type ContinuousAgent = z.infer<typeof continuousAgentSchema>;
export type CreateAgentRequest = z.infer<typeof createAgentRequestSchema>;
export type UpdateAgentRequest = z.infer<typeof updateAgentRequestSchema>;
export type AgentExecutionLog = z.infer<typeof agentExecutionLogSchema>;




