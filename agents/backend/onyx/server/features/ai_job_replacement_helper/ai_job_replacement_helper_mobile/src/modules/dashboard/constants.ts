export const QUICK_ACTIONS = [
  {
    id: 'search-jobs',
    icon: 'search',
    label: 'Search Jobs',
    route: '/(tabs)/jobs',
  },
  {
    id: 'analyze-cv',
    icon: 'document-text',
    label: 'Analyze CV',
    route: '/cv-analyzer',
  },
  {
    id: 'ai-mentor',
    icon: 'chatbubbles',
    label: 'AI Mentor',
    route: '/mentoring',
  },
  {
    id: 'practice-interview',
    icon: 'videocam',
    label: 'Practice Interview',
    route: '/interview',
  },
] as const;

export const DASHBOARD_REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes


