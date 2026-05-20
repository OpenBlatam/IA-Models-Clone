import type { ContinuousAgent } from "../../../types";
import type { AgentEntity } from "../../../domain/entities/Agent";

/**
 * Factory para crear handlers optimizados
 * Pattern: Factory para crear funciones con closure optimizado
 */

export function createToggleHandler(
  entityMap: Map<string, AgentEntity>,
  toggleFn: (entity: AgentEntity) => void
) {
  return (agent: ContinuousAgent) => {
    const entity = entityMap.get(agent.id);
    if (entity) {
      toggleFn(entity);
    }
  };
}

export function createFilterHandler<T>(
  filterFn: (value: T) => void
): (value: T) => void {
  return filterFn;
}








