import type { IAgentRepository } from "../../domain/repositories/IAgentRepository";
import type { AgentEntity } from "../../domain/entities/Agent";
import { Result } from "../../utils/result";

export class GetAgentsUseCase {
  constructor(private readonly repository: IAgentRepository) {}

  async execute(): Promise<Result<AgentEntity[], Error>> {
    // El repositorio ahora retorna Result directamente
    // Mantenemos la interfaz para compatibilidad futura
    const result = await this.repository.findAll();
    return result;
  }
}
