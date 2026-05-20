type ClassValue = string | number | boolean | undefined | null | Record<string, boolean> | ClassValue[];

export function classNames(...classes: ClassValue[]): string {
  const result: string[] = [];

  classes.forEach((cls) => {
    if (!cls) return;

    if (typeof cls === 'string' || typeof cls === 'number') {
      result.push(String(cls));
    } else if (Array.isArray(cls)) {
      const inner = classNames(...cls);
      if (inner) result.push(inner);
    } else if (typeof cls === 'object') {
      Object.entries(cls).forEach(([key, value]) => {
        if (value) result.push(key);
      });
    }
  });

  return result.join(' ');
}

export function cn(...classes: ClassValue[]): string {
  return classNames(...classes);
}

