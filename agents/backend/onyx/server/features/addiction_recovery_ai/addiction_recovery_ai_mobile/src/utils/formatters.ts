// Pure functions for formatting

export function formatDate(date: Date, format: string = "EEEE, d 'de' MMMM"): string {
  // This would use date-fns in real implementation
  return date.toLocaleDateString('es-ES', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

export function formatNumber(value: number, decimals: number = 0): string {
  return value.toFixed(decimals);
}

export function formatPercentage(value: number, decimals: number = 0): string {
  return `${formatNumber(value, decimals)}%`;
}

export function capitalizeFirst(str: string): string {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

