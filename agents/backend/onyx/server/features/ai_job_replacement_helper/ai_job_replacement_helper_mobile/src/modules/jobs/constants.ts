import { Dimensions } from 'react-native';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export const CARD_WIDTH = SCREEN_WIDTH - 40;
export const SWIPE_THRESHOLD = 100;

export const JOB_ACTIONS = {
  LIKE: 'like',
  DISLIKE: 'dislike',
  SAVE: 'save',
  APPLY: 'apply',
} as const;

export const JOB_FILTERS = {
  JOB_TYPES: ['full-time', 'part-time', 'contract', 'internship', 'remote'],
  EXPERIENCE_LEVELS: ['entry', 'mid', 'senior', 'executive'],
} as const;


