import type { ContinuousAgent, FilterStatus } from "../types";

export function filterAgents(
  agents: ContinuousAgent[],
  searchQuery: string,
  filterStatus: FilterStatus
): ContinuousAgent[] {
  let items = agents;

  // Filtrar por estado
  if (filterStatus === "active") {
    items = items.filter((a) => a.isActive);
  } else if (filterStatus === "inactive") {
    items = items.filter((a) => !a.isActive);
  }

  // Filtrar por búsqueda
  if (searchQuery.trim()) {
    const q = searchQuery.toLowerCase();
    items = items.filter(
      (a) =>
        a.name.toLowerCase().includes(q) ||
        a.description?.toLowerCase().includes(q) ||
        a.config.taskType.toLowerCase().includes(q)
    );
  }

  return items;
}








