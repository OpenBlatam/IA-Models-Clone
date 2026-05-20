export const STEP_CATEGORIES = [
  'evaluation',
  'skills',
  'learning',
  'networking',
  'application',
  'interview',
  'motivation',
] as const;

export const STEP_STATUS = {
  NOT_STARTED: 'not_started',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
} as const;

export const STEP_POINTS = {
  START: 10,
  COMPLETE: 25,
} as const;


