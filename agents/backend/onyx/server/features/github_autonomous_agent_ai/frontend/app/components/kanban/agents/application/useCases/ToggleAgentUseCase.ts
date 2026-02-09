import type { IAgentRepository } from "../../domain/repositories/IAgentRepository";
import type { AgentEntity } from "../../domain/entities/Agent";
import { AgentValidationError } from "../../domain/errors/AgentErrors";
import { Result } from "../../utils/result";

export class ToggleAgentUseCase {
  constructor(private readonly repository: IAgentRepository) {}

  async execute(agent: AgentEntity): Promise<Result<AgentEntity, Error>> {
    // Validación de negocio
    if (agent.isActive && !agent.canBePaused()) {
      return Result.err(
        new AgentValidationError("Agent cannot be paused", "isActive")
      );
    }

    if (!agent.isActive && !agent.canBeActivated()) {
      return Result.err(
        new AgentValidationError("Agent cannot be activated", "isActive")
      );
    }

    const result = await this.repository.toggleActive(agent.id, !agent.isActive);
    return result;
  }
}
