export const trimFormValues = <T extends Record<string, unknown>>(values: T): T => {
  const trimmed: Record<string, unknown> = {};

  Object.entries(values).forEach(([key, value]) => {
    if (typeof value === 'string') {
      trimmed[key] = value.trim();
    } else {
      trimmed[key] = value;
    }
  });

  return trimmed as T;
};

export const hasEmptyRequiredFields = <T extends Record<string, unknown>>(
  values: T,
  requiredFields: (keyof T)[]
): boolean => {
  return requiredFields.some((field) => {
    const value = values[field];
    if (typeof value === 'string') {
      return value.trim().length === 0;
    }
    return value === null || value === undefined;
  });
};

export const getEmptyRequiredFields = <T extends Record<string, unknown>>(
  values: T,
  requiredFields: (keyof T)[]
): string[] => {
  return requiredFields
    .filter((field) => {
      const value = values[field];
      if (typeof value === 'string') {
        return value.trim().length === 0;
      }
      return value === null || value === undefined;
    })
    .map((field) => String(field));
};



