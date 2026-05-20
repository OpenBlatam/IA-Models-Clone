export function memoize<Args extends any[], Return>(
  fn: (...args: Args) => Return
): (...args: Args) => Return {
  const cache = new Map<string, Return>();

  return (...args: Args): Return => {
    const key = JSON.stringify(args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
}

export function memoizeWithTTL<Args extends any[], Return>(
  fn: (...args: Args) => Return,
  ttl: number
): (...args: Args) => Return {
  const cache = new Map<
    string,
    { value: Return; expires: number }
  >();

  return (...args: Args): Return => {
    const key = JSON.stringify(args);
    const now = Date.now();

    const cached = cache.get(key);
    if (cached && cached.expires > now) {
      return cached.value;
    }

    const result = fn(...args);
    cache.set(key, {
      value: result,
      expires: now + ttl,
    });

    // Cleanup expired entries
    for (const [k, v] of cache.entries()) {
      if (v.expires <= now) {
        cache.delete(k);
      }
    }

    return result;
  };
}

