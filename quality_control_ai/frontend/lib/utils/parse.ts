export const parseJSON = <T,>(json: string, defaultValue?: T): T | null => {
  try {
    return JSON.parse(json) as T;
  } catch {
    return defaultValue ?? null;
  }
};

export const parseNumber = (value: string | number, defaultValue = 0): number => {
  if (typeof value === 'number') return value;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
};

export const parseInteger = (value: string | number, defaultValue = 0): number => {
  if (typeof value === 'number') return Math.floor(value);
  const parsed = parseInt(value, 10);
  return isNaN(parsed) ? defaultValue : parsed;
};

export const parseBoolean = (value: string | boolean | number): boolean => {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  const lower = value.toLowerCase().trim();
  return lower === 'true' || lower === '1' || lower === 'yes' || lower === 'on';
};

export const parseDate = (value: string | Date, defaultValue?: Date): Date | null => {
  if (value instanceof Date) return value;
  const parsed = new Date(value);
  return isNaN(parsed.getTime()) ? (defaultValue ?? null) : parsed;
};

export const parseCSV = (csv: string, delimiter = ','): string[][] => {
  const lines = csv.split('\n');
  return lines
    .filter((line) => line.trim())
    .map((line) => {
      const values: string[] = [];
      let current = '';
      let inQuotes = false;

      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];

        if (char === '"') {
          if (inQuotes && nextChar === '"') {
            current += '"';
            i++;
          } else {
            inQuotes = !inQuotes;
          }
        } else if (char === delimiter && !inQuotes) {
          values.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }

      values.push(current.trim());
      return values;
    });
};

export const parseQueryString = (queryString: string): Record<string, string> => {
  const params: Record<string, string> = {};
  const searchParams = new URLSearchParams(queryString);
  searchParams.forEach((value, key) => {
    params[key] = value;
  });
  return params;
};

export const parseFormData = (form: HTMLFormElement): Record<string, string> => {
  const formData = new FormData(form);
  const data: Record<string, string> = {};
  formData.forEach((value, key) => {
    data[key] = value.toString();
  });
  return data;
};

