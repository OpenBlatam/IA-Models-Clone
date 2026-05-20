import type { AgentEntity } from "../../domain/entities/Agent";
import type { ContinuousAgent } from "../../types";
import { container } from "../../infrastructure/container/DIContainer";

/**
 * Selectors para derivar estado de agentes
 * Pattern: Selector functions para evitar cálculos duplicados
 */

export const selectAgentDTOs = (entities: AgentEntity[]): ContinuousAgent[] => {
  if (!Array.isArray(entities)) {
    console.warn("selectAgentDTOs received non-array value:", entities);
    return [];
  }
  return container.adapter.toDTOArray(entities);
};

export const selectAgentById = (entities: AgentEntity[], id: string): AgentEntity | undefined => {
  if (!Array.isArray(entities)) return undefined;
  return entities.find((e) => e.id === id);
};

export const selectActiveAgents = (entities: AgentEntity[]): AgentEntity[] => {
  if (!Array.isArray(entities)) return [];
  return entities.filter((e) => e.isActive);
};

export const selectInactiveAgents = (entities: AgentEntity[]): AgentEntity[] => {
  if (!Array.isArray(entities)) return [];
  return entities.filter((e) => !e.isActive);
};

export const selectAgentCount = (entities: AgentEntity[]): number => {
  if (!Array.isArray(entities)) return 0;
  return entities.length;
};

export const createEntityMap = (entities: AgentEntity[]): Map<string, AgentEntity> => {
  if (!Array.isArray(entities)) return new Map();
  return new Map(entities.map((e) => [e.id, e]));
};





