'use client';

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { useModels } from '@/lib/hooks/use-manuals';
import type { ModelSelectProps } from '@/lib/types/components';

export const ModelSelect = ({
  value,
  onValueChange,
  placeholder = 'Modelo por defecto',
}: ModelSelectProps): JSX.Element => {
  const { data: modelsData } = useModels();

  return (
    <Select
      value={value}
      onValueChange={onValueChange}
    >
      <SelectTrigger>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {modelsData?.models.map((model) => (
          <SelectItem key={model.id} value={model.id}>
            {model.name || model.id}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

