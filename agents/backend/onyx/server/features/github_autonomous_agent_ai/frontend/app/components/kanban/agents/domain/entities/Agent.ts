import { z } from "zod";

// Schema de validación
export const AgentStatsSchema = z.object({
  totalExecutions: z.number().min(0),
  successfulExecutions: z.number().min(0),
  failedExecutions: z.number().min(0),
  creditsUsed: z.number().min(0),
  lastExecutionAt: z.string().nullable().optional(),
  nextExecutionAt: z.string().nullable().optional(),
  averageExecutionTime: z.number().min(0).optional(),
});

export const AgentConfigSchema = z.object({
  taskType: z.enum([
    "content_generation",
    "data_processing",
    "api_monitoring",
    "automated_research",
    "custom",
  ]),
  frequency: z.number().min(0),
  parameters: z.record(z.unknown()),
  goal: z.string().optional(),
});

export const AgentSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  description: z.string().optional(),
  isActive: z.boolean(),
  config: AgentConfigSchema,
  stats: AgentStatsSchema,
  createdAt: z.string(),
  updatedAt: z.string(),
  stripeCreditsRemaining: z.number().nullable().optional(),
});

// Tipos inferidos
export type AgentStats = z.infer<typeof AgentStatsSchema>;
export type AgentConfig = z.infer<typeof AgentConfigSchema>;
export type Agent = z.infer<typeof AgentSchema>;

// Entidad de dominio con lógica de negocio
export class AgentEntity {
  constructor(private readonly data: Agent) {}

  get id(): string {
    return this.data.id;
  }

  get name(): string {
    return this.data.name;
  }

  get isActive(): boolean {
    return this.data.isActive;
  }

  get stats(): AgentStats {
    return this.data.stats;
  }

  get config(): AgentConfig {
    return this.data.config;
  }

  // Lógica de negocio
  canBeActivated(): boolean {
    return !this.data.isActive;
  }

  canBePaused(): boolean {
    return this.data.isActive;
  }

  getSuccessRate(): number {
    const { totalExecutions, successfulExecutions } = this.data.stats;
    if (totalExecutions === 0) return 0;
    return (successfulExecutions / totalExecutions) * 100;
  }

  hasRecentActivity(): boolean {
    if (!this.data.stats.lastExecutionAt) return false;
    const lastExec = new Date(this.data.stats.lastExecutionAt);
    const now = new Date();
    const diffMinutes = (now.getTime() - lastExec.getTime()) / 1000 / 60;
    return diffMinutes < 5;
  }

  toPlainObject(): Agent {
    return { ...this.data };
  }

  static fromPlainObject(data: unknown): AgentEntity {
    const validated = AgentSchema.parse(data);
    return new AgentEntity(validated);
  }
}








