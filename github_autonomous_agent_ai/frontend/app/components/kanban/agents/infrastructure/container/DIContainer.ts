import { AgentAdapter } from "../adapters/AgentAdapter";
import { AgentRepository } from "../repositories/AgentRepository";
import { GetAgentsUseCase } from "../../application/useCases/GetAgentsUseCase";
import { ToggleAgentUseCase } from "../../application/useCases/ToggleAgentUseCase";

/**
 * Dependency Injection Container
 * Singleton pattern para evitar múltiples instancias
 */
class DIContainer {
  private static instance: DIContainer;
  
  private _adapter: AgentAdapter;
  private _repository: AgentRepository;
  private _getAgentsUseCase: GetAgentsUseCase;
  private _toggleAgentUseCase: ToggleAgentUseCase;

  private constructor() {
    this._adapter = new AgentAdapter();
    this._repository = new AgentRepository(this._adapter);
    this._getAgentsUseCase = new GetAgentsUseCase(this._repository);
    this._toggleAgentUseCase = new ToggleAgentUseCase(this._repository);
  }

  static getInstance(): DIContainer {
    if (!DIContainer.instance) {
      DIContainer.instance = new DIContainer();
    }
    return DIContainer.instance;
  }

  get adapter(): AgentAdapter {
    return this._adapter;
  }

  get repository(): AgentRepository {
    return this._repository;
  }

  get getAgentsUseCase(): GetAgentsUseCase {
    return this._getAgentsUseCase;
  }

  get toggleAgentUseCase(): ToggleAgentUseCase {
    return this._toggleAgentUseCase;
  }
}

export const container = DIContainer.getInstance();








