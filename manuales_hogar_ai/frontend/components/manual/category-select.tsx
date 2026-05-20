'use client';

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { useCategories } from '@/lib/hooks/use-manuals';
import type { Category } from '@/lib/types/api';
import type { CategorySelectProps } from '@/lib/types/components';

export const CategorySelect = ({
  value,
  onValueChange,
  placeholder = 'Selecciona una categoría',
  includeAll = false,
}: CategorySelectProps): JSX.Element => {
  const { data: categoriesData } = useCategories();

  const handleValueChange = (newValue: string): void => {
    if (newValue === 'all') {
      onValueChange(undefined);
    } else {
      onValueChange(newValue as Category);
    }
  };

  return (
    <Select
      value={value || (includeAll ? 'all' : undefined)}
      onValueChange={handleValueChange}
    >
      <SelectTrigger>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {includeAll && <SelectItem value="all">Todas las categorías</SelectItem>}
        {categoriesData?.categories.map((cat) => (
          <SelectItem key={cat} value={cat}>
            {categoriesData.category_names[cat as keyof typeof categoriesData.category_names]}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

