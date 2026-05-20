import { useMemo } from 'react';
import { validateRequired, validateMinLength, validateMaxLength, validateEmail, validateUsername } from '@/lib/utils';

interface ValidationRule {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  email?: boolean;
  username?: boolean;
  custom?: (value: string) => string | null;
}

interface ValidationRules {
  [key: string]: ValidationRule;
}

export const useFormValidation = <T extends Record<string, string>>(
  rules: ValidationRules
) => {
  const validate = useMemo(
    () => (values: T): Record<string, string> => {
      const errors: Record<string, string> = {};

      Object.entries(rules).forEach(([field, rule]) => {
        const value = values[field] || '';

        if (rule.required && !validateRequired(value)) {
          errors[field] = 'This field is required';
          return;
        }

        if (value && rule.minLength && !validateMinLength(value, rule.minLength)) {
          errors[field] = `Must be at least ${rule.minLength} characters`;
          return;
        }

        if (value && rule.maxLength && !validateMaxLength(value, rule.maxLength)) {
          errors[field] = `Must be at most ${rule.maxLength} characters`;
          return;
        }

        if (value && rule.email && !validateEmail(value)) {
          errors[field] = 'Please enter a valid email address';
          return;
        }

        if (value && rule.username && !validateUsername(value)) {
          errors[field] = 'Invalid username format';
          return;
        }

        if (value && rule.custom) {
          const customError = rule.custom(value);
          if (customError) {
            errors[field] = customError;
          }
        }
      });

      return errors;
    },
    [rules]
  );

  return { validate };
};



