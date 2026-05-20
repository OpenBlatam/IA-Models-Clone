export const queryKeys = {
  identities: {
    all: ['identities'] as const,
    lists: () => [...queryKeys.identities.all, 'list'] as const,
    list: (filters: string) => [...queryKeys.identities.lists(), { filters }] as const,
    details: () => [...queryKeys.identities.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.identities.details(), id] as const,
    generatedContent: (id: string) => [...queryKeys.identities.detail(id), 'generated-content'] as const,
    analytics: (id: string) => [...queryKeys.identities.detail(id), 'analytics'] as const,
    versions: (id: string) => [...queryKeys.identities.detail(id), 'versions'] as const,
  },
  templates: {
    all: ['templates'] as const,
    lists: () => [...queryKeys.templates.all, 'list'] as const,
    detail: (id: string) => [...queryKeys.templates.all, 'detail', id] as const,
  },
  tasks: {
    all: ['tasks'] as const,
    lists: () => [...queryKeys.tasks.all, 'list'] as const,
    detail: (id: string) => [...queryKeys.tasks.all, 'detail', id] as const,
  },
  alerts: {
    all: ['alerts'] as const,
    lists: () => [...queryKeys.alerts.all, 'list'] as const,
  },
  dashboard: {
    all: ['dashboard'] as const,
  },
  metrics: {
    all: ['metrics'] as const,
  },
  analytics: {
    all: ['analytics'] as const,
    stats: () => [...queryKeys.analytics.all, 'stats'] as const,
    trends: () => [...queryKeys.analytics.all, 'trends'] as const,
  },
  notifications: {
    all: ['notifications'] as const,
  },
  recommendations: {
    all: ['recommendations'] as const,
    system: () => [...queryKeys.recommendations.all, 'system'] as const,
    identity: (id: string) => [...queryKeys.recommendations.all, 'identity', id] as const,
  },
  schedules: {
    all: ['schedules'] as const,
  },
  abTests: {
    all: ['ab-tests'] as const,
    detail: (id: string) => [...queryKeys.abTests.all, 'detail', id] as const,
    winner: (id: string) => [...queryKeys.abTests.detail(id), 'winner'] as const,
  },
  backups: {
    all: ['backups'] as const,
  },
  plugins: {
    all: ['plugins'] as const,
    detail: (id: string) => [...queryKeys.plugins.all, 'detail', id] as const,
  },
  health: {
    all: ['health'] as const,
  },
} as const;



