import { AgentEntity } from "../../domain/entities/Agent";
import type { ContinuousAgent } from "../../types";

export class AgentAdapter {
  toDomain(data: unknown): AgentEntity {
    try {
      return AgentEntity.fromPlainObject(data);
    } catch (error) {
      console.error("Error adapting agent data", error);
      throw new Error("Invalid agent data format");
    }
  }

  toDTO(entity: AgentEntity): ContinuousAgent {
    const agent = entity.toPlainObject();
    return {
      id: agent.id,
      name: agent.name,
      description: agent.description,
      isActive: agent.isActive,
      config: {
        taskType: agent.config.taskType,
      },
      stats: {
        totalExecutions: agent.stats.totalExecutions,
        successfulExecutions: agent.stats.successfulExecutions,
        failedExecutions: agent.stats.failedExecutions,
        creditsUsed: agent.stats.creditsUsed,
        lastExecutionAt: agent.stats.lastExecutionAt ?? null,
      },
      createdAt: agent.createdAt,
    };
  }

  toDTOArray(entities: AgentEntity[]): ContinuousAgent[] {
    if (!Array.isArray(entities)) {
      console.warn("toDTOArray received non-array value:", entities);
      return [];
    }
    return entities.map((entity) => this.toDTO(entity));
  }
}





