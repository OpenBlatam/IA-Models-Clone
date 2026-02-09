import type { AgentEntity } from "../entities/Agent";

export interface IAgentRepository {
  findAll(): Promise<AgentEntity[]>;
  findById(id: string): Promise<AgentEntity | null>;
  toggleActive(id: string, isActive: boolean): Promise<AgentEntity>;
}

export interface IAgentRepositoryResult<T> {
  success: boolean;
  data?: T;
  error?: Error;
}








