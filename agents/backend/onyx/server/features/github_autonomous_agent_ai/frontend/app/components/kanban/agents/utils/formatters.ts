export function formatSuccessRate(rate: number): string {
  return `${rate.toFixed(1)}%`;
}

export function formatCredits(credits: number): string {
  return credits.toFixed(2);
}

export function formatDate(dateString: string | null | undefined): string {
  if (!dateString) return "-";
  return new Date(dateString).toLocaleString();
}

export function formatAgentCount(count: number): string {
  return `${count} agente${count !== 1 ? "s" : ""} configurado${count !== 1 ? "s" : ""}`;
}








