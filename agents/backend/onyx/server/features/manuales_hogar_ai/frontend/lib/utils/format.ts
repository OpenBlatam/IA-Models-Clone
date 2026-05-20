import { format as formatDate } from 'date-fns';

export const formatManualDate = (date: string | Date): string => {
  return formatDate(new Date(date), 'PP');
};

export const formatCategoryName = (category: string): string => {
  return category.charAt(0).toUpperCase() + category.slice(1);
};

