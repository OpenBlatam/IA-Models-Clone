import { sortByValue, extractModelName } from './data-transform';
import type { StatisticsResponse } from '../types/api';

export const getTopModel = (stats: StatisticsResponse): string => {
  if (!stats.top_models || stats.top_models.length === 0) {
    return 'N/A';
  }
  return extractModelName(stats.top_models[0].model);
};

export const getTopModelUsage = (stats: StatisticsResponse): number => {
  return stats.top_models[0]?.count || 0;
};

export const getSortedCategories = (stats: StatisticsResponse): Array<[string, number]> => {
  return sortByValue(stats.category_stats) as Array<[string, number]>;
};

export const getCategoryCount = (stats: StatisticsResponse): number => {
  return Object.keys(stats.category_stats).length;
};

