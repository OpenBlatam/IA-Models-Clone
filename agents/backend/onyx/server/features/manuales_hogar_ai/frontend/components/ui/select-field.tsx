'use client';

import { FormField } from './form-field';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './select';
import type { CategorySelectProps, ModelSelectProps } from '@/lib/types/components';
import { useCategories } from '@/lib/hooks/use-manuals';
import { useModels } from '@/lib/hooks/use-manuals';
import type { Category } from '@/lib/types/api';

interface SelectFieldProps {
  label: string;
  htmlFor: string;
  value?: string;
  onValueChange: (value: string | undefined) => void;
  placeholder?: string;
  options: Array<{ value: string; label: string }>;
}

export const SelectField = ({
  label,
  htmlFor,
  value,
  onValueChange,
  placeholder,
  options,
}: SelectFieldProps): JSX.Element => {
  return (
    <FormField label={label} htmlFor={htmlFor}>
      <Select value={value} onValueChange={onValueChange}>
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </FormField>
  );
};

interface CategorySelectFieldProps extends Omit<CategorySelectProps, 'value' | 'onValueChange'> {
  label: string;
  htmlFor: string;
  value?: Category | string;
  onValueChange: (value: Category | string | undefined) => void;
}

export const CategorySelectField = ({
  label,
  htmlFor,
  value,
  onValueChange,
  placeholder = 'Selecciona una categoría',
  includeAll = false,
}: CategorySelectFieldProps): JSX.Element => {
  const { data: categoriesData } = useCategories();

  const handleValueChange = (newValue: string): void => {
    if (newValue === 'all') {
      onValueChange(undefined);
    } else {
      onValueChange(newValue as Category);
    }
  };

  const options = [
    ...(includeAll ? [{ value: 'all', label: 'Todas las categorías' }] : []),
    ...(categoriesData?.categories.map((cat) => ({
      value: cat,
      label: categoriesData.category_names[cat as keyof typeof categoriesData.category_names],
    })) || []),
  ];

  return (
    <SelectField
      label={label}
      htmlFor={htmlFor}
      value={value || (includeAll ? 'all' : undefined)}
      onValueChange={handleValueChange}
      placeholder={placeholder}
      options={options}
    />
  );
};

interface ModelSelectFieldProps extends Omit<ModelSelectProps, 'value' | 'onValueChange'> {
  label: string;
  htmlFor: string;
  value?: string;
  onValueChange: (value: string | undefined) => void;
}

export const ModelSelectField = ({
  label,
  htmlFor,
  value,
  onValueChange,
  placeholder = 'Selecciona un modelo',
}: ModelSelectFieldProps): JSX.Element => {
  const { data: modelsData } = useModels();

  const options =
    modelsData?.models.map((model) => ({
      value: model.id,
      label: model.name || model.id,
    })) || [];

  return (
    <SelectField
      label={label}
      htmlFor={htmlFor}
      value={value}
      onValueChange={onValueChange}
      placeholder={placeholder}
      options={options}
    />
  );
};

