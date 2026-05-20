export const GENERATOR_TABS = {
  TEXT: 'text',
  IMAGE: 'image',
  COMBINED: 'combined',
} as const;

export const SEARCH_TABS = {
  SIMPLE: 'simple',
  SEMANTIC: 'semantic',
  ADVANCED: 'advanced',
} as const;

export type GeneratorTab = typeof GENERATOR_TABS[keyof typeof GENERATOR_TABS];
export type SearchTab = typeof SEARCH_TABS[keyof typeof SEARCH_TABS];

