export const queryKeys = {
  dashboard: (artistId: string) => ['dashboard', artistId] as const,
  dashboardSummary: (artistId: string, date: string) => ['dashboard', artistId, 'summary', date] as const,
  
  events: (artistId: string, params?: Record<string, any>) => ['events', artistId, params] as const,
  event: (artistId: string, eventId: string) => ['events', artistId, eventId] as const,
  wardrobeRecommendation: (artistId: string, eventId: string) => ['wardrobe-recommendation', artistId, eventId] as const,
  
  routines: (artistId: string) => ['routines', artistId] as const,
  routine: (artistId: string, routineId: string) => ['routines', artistId, routineId] as const,
  pendingRoutines: (artistId: string) => ['routines', artistId, 'pending'] as const,
  
  protocols: (artistId: string) => ['protocols', artistId] as const,
  protocol: (artistId: string, protocolId: string) => ['protocols', artistId, protocolId] as const,
  protocolCompliance: (artistId: string, protocolId: string) => ['protocol-compliance', artistId, protocolId] as const,
  
  wardrobeItems: (artistId: string) => ['wardrobe-items', artistId] as const,
  wardrobeItem: (artistId: string, itemId: string) => ['wardrobe-items', artistId, itemId] as const,
  outfits: (artistId: string) => ['outfits', artistId] as const,
  outfit: (artistId: string, outfitId: string) => ['outfits', artistId, outfitId] as const,
  
  alerts: (artistId: string) => ['alerts', artistId] as const,
  eventSearch: (artistId: string, query: string) => ['event-search', artistId, query] as const,
};

