export const diff = <T,>(oldObj: T, newObj: T): Partial<T> => {
  const changes: Partial<T> = {};
  const keys = new Set([...Object.keys(oldObj as Record<string, unknown>), ...Object.keys(newObj as Record<string, unknown>)]);

  keys.forEach((key) => {
    const oldValue = (oldObj as Record<string, unknown>)[key];
    const newValue = (newObj as Record<string, unknown>)[key];

    if (oldValue !== newValue) {
      (changes as Record<string, unknown>)[key] = newValue;
    }
  });

  return changes;
};

export const deepDiff = <T extends Record<string, unknown>>(
  oldObj: T,
  newObj: T
): Partial<T> => {
  const changes: Partial<T> = {};
  const keys = new Set([...Object.keys(oldObj), ...Object.keys(newObj)]);

  keys.forEach((key) => {
    const oldValue = oldObj[key];
    const newValue = newObj[key];

    if (oldValue !== newValue) {
      if (
        typeof oldValue === 'object' &&
        oldValue !== null &&
        typeof newValue === 'object' &&
        newValue !== null &&
        !Array.isArray(oldValue) &&
        !Array.isArray(newValue)
      ) {
        const nestedDiff = deepDiff(
          oldValue as Record<string, unknown>,
          newValue as Record<string, unknown>
        );
        if (Object.keys(nestedDiff).length > 0) {
          (changes as Record<string, unknown>)[key] = nestedDiff;
        }
      } else {
        (changes as Record<string, unknown>)[key] = newValue;
      }
    }
  });

  return changes;
};

export const hasChanges = <T,>(oldObj: T, newObj: T): boolean => {
  const changes = diff(oldObj, newObj);
  return Object.keys(changes).length > 0;
};

export const hasDeepChanges = <T extends Record<string, unknown>>(
  oldObj: T,
  newObj: T
): boolean => {
  const changes = deepDiff(oldObj, newObj);
  return Object.keys(changes).length > 0;
};

