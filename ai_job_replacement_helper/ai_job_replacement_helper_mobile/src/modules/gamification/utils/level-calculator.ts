import type { LevelProgress } from '../types';

const XP_PER_LEVEL = [
  100, 250, 500, 1000, 2000, 3500, 5500, 8000, 12000, 20000,
];

export function calculateLevel(xp: number): number {
  let level = 1;
  let totalXP = 0;

  for (let i = 0; i < XP_PER_LEVEL.length; i++) {
    totalXP += XP_PER_LEVEL[i];
    if (xp >= totalXP) {
      level = i + 2;
    } else {
      break;
    }
  }

  return Math.min(level, 10);
}

export function getXPForLevel(level: number): number {
  if (level <= 1) return 0;
  if (level > 10) return XP_PER_LEVEL.reduce((sum, xp) => sum + xp, 0);

  return XP_PER_LEVEL.slice(0, level - 1).reduce((sum, xp) => sum + xp, 0);
}

export function getXPToNextLevel(currentXP: number, currentLevel: number): number {
  const xpForNextLevel = getXPForLevel(currentLevel + 1);
  return xpForNextLevel - currentXP;
}

export function calculateLevelProgress(xp: number): LevelProgress {
  const currentLevel = calculateLevel(xp);
  const xpToNextLevel = getXPToNextLevel(xp, currentLevel);
  const xpForCurrentLevel = getXPForLevel(currentLevel);
  const xpForNextLevel = getXPForLevel(currentLevel + 1);
  const progressXP = xp - xpForCurrentLevel;
  const levelRangeXP = xpForNextLevel - xpForCurrentLevel;
  const progressPercentage = levelRangeXP > 0 ? (progressXP / levelRangeXP) * 100 : 0;

  return {
    currentLevel,
    currentXP: xp,
    xpToNextLevel,
    progressPercentage: Math.min(progressPercentage, 100),
  };
}


