/**
 * Formats the agent count text with proper pluralization
 * @param count - The number of agents
 * @returns Formatted string with proper pluralization
 */
export const getAgentCountText = (count: number): string => {
  if (count === 0) {
    return "No hay agentes configurados";
  }
  const plural = count !== 1;
  return `${count} agente${plural ? "s" : ""} configurado${plural ? "s" : ""}`;
};



