'use client';

import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Plus, Trash2 } from 'lucide-react';

interface FieldArrayInputProps {
  fields: { id: string }[];
  register: (index: number) => any;
  onAppend: () => void;
  onRemove: (index: number) => void;
  label: string;
  placeholder?: string;
  error?: string;
  required?: boolean;
  minFields?: number;
}

const FieldArrayInput = ({
  fields,
  register,
  onAppend,
  onRemove,
  label,
  placeholder,
  error,
  required,
  minFields = 1,
}: FieldArrayInputProps) => {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      {fields.map((field, index) => (
        <div key={field.id} className="flex gap-2 mb-2">
          <Input
            {...register(index)}
            placeholder={placeholder}
            className="flex-1"
            error={error}
          />
          {fields.length > minFields && (
            <Button
              type="button"
              variant="danger"
              size="sm"
              onClick={() => onRemove(index)}
              aria-label="Eliminar"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          )}
        </div>
      ))}
      <Button type="button" variant="secondary" size="sm" onClick={onAppend} className="mt-2">
        <Plus className="w-4 h-4 mr-2" />
        Agregar
      </Button>
      {error && typeof error === 'string' && (
        <p className="mt-1 text-sm text-red-600" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};

export { FieldArrayInput };

