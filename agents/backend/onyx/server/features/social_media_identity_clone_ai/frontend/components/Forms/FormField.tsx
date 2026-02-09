import { memo } from 'react';
import Input from '@/components/UI/Input';
import Select from '@/components/UI/Select';
import Textarea from '@/components/UI/Textarea';
import { cn } from '@/lib/utils';

interface FormFieldProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'select' | 'textarea';
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  error?: string;
  required?: boolean;
  placeholder?: string;
  options?: Array<{ value: string; label: string }>;
  className?: string;
  disabled?: boolean;
}

const FormField = memo(({
  type = 'text',
  label,
  value,
  onChange,
  error,
  required = false,
  placeholder,
  options,
  className = '',
  disabled = false,
}: FormFieldProps): JSX.Element => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>): void => {
    onChange(e.target.value);
  };

  if (type === 'select' && options) {
    return (
      <div className={cn('space-y-1', className)}>
        <Select
          label={label}
          value={String(value)}
          onChange={handleChange}
          options={options}
          required={required}
          disabled={disabled}
        />
        {error && <p className="text-sm text-red-600" role="alert">{error}</p>}
      </div>
    );
  }

  if (type === 'textarea') {
    return (
      <div className={cn('space-y-1', className)}>
        <Textarea
          label={label}
          value={String(value)}
          onChange={handleChange}
          required={required}
          placeholder={placeholder}
          disabled={disabled}
        />
        {error && <p className="text-sm text-red-600" role="alert">{error}</p>}
      </div>
    );
  }

  return (
    <div className={cn('space-y-1', className)}>
      <Input
        type={type}
        label={label}
        value={String(value)}
        onChange={handleChange}
        required={required}
        placeholder={placeholder}
        disabled={disabled}
      />
      {error && <p className="text-sm text-red-600" role="alert">{error}</p>}
    </div>
  );
});

FormField.displayName = 'FormField';

export default FormField;



