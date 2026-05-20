/**
 * Network utilities
 */

// Check if online
export function isOnline(): boolean {
  if (typeof window === 'undefined') {
    return true;
  }
  return navigator.onLine;
}

// Check connection type
export function getConnectionType(): string | null {
  if (typeof window === 'undefined' || !('connection' in navigator)) {
    return null;
  }

  const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;
  
  if (!connection) {
    return null;
  }

  return connection.effectiveType || connection.type || 'unknown';
}

// Get connection speed
export function getConnectionSpeed(): string | null {
  if (typeof window === 'undefined' || !('connection' in navigator)) {
    return null;
  }

  const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;
  
  if (!connection) {
    return null;
  }

  return connection.effectiveType || 'unknown';
}

// Check if connection is slow
export function isSlowConnection(): boolean {
  const speed = getConnectionSpeed();
  return speed === 'slow-2g' || speed === '2g' || speed === '3g';
}

// Ping URL
export async function ping(url: string, timeout: number = 5000): Promise<number> {
  const startTime = performance.now();

  try {
    await fetch(url, {
      method: 'HEAD',
      mode: 'no-cors',
      cache: 'no-cache',
    });
    
    return performance.now() - startTime;
  } catch {
    return -1;
  }
}

// Check if URL is reachable
export async function isReachable(url: string, timeout: number = 5000): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(url, {
      method: 'HEAD',
      mode: 'no-cors',
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    return true;
  } catch {
    return false;
  }
}

// Get network status
export function getNetworkStatus(): {
  online: boolean;
  type: string | null;
  speed: string | null;
  slow: boolean;
} {
  return {
    online: isOnline(),
    type: getConnectionType(),
    speed: getConnectionSpeed(),
    slow: isSlowConnection(),
  };
}



