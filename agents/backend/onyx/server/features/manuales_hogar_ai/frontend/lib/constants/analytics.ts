export const ANALYTICS_PERIODS = [
  { value: 7, label: '7 días' },
  { value: 30, label: '30 días' },
  { value: 90, label: '90 días' },
  { value: 365, label: '1 año' },
] as const;

export const ANALYTICS_INTERVALS = [
  { value: 'day' as const, label: 'Diario' },
  { value: 'week' as const, label: 'Semanal' },
] as const;

