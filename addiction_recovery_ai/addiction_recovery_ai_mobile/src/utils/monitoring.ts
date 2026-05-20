import * as Sentry from '@sentry/react-native';

export interface PerformanceMetric {
  name: string;
  value: number;
  unit?: string;
  tags?: Record<string, string>;
}

export function trackPerformance(metric: PerformanceMetric): void {
  try {
    if (Sentry.metrics && typeof Sentry.metrics.distribution === 'function') {
      Sentry.metrics.distribution(metric.name, metric.value, {
        unit: metric.unit || 'millisecond',
        tags: metric.tags,
      });
    }
  } catch (error) {
    console.warn('Failed to track performance metric:', error);
  }
}

export function trackEvent(
  name: string,
  properties?: Record<string, unknown>
): void {
  try {
    Sentry.addBreadcrumb({
      category: 'user',
      message: name,
      data: properties,
      level: 'info',
    });
  } catch (error) {
    console.warn('Failed to track event:', error);
  }
}

export function setUserContext(user: {
  id: string;
  email?: string;
  username?: string;
}): void {
  try {
    Sentry.setUser({
      id: user.id,
      email: user.email,
      username: user.username,
    });
  } catch (error) {
    console.warn('Failed to set user context:', error);
  }
}

export function clearUserContext(): void {
  try {
    Sentry.setUser(null);
  } catch (error) {
    console.warn('Failed to clear user context:', error);
  }
}

export function startTransaction(name: string, op = 'navigation') {
  try {
    if (typeof Sentry.startTransaction === 'function') {
      return Sentry.startTransaction({
        name,
        op,
      });
    }
  } catch (error) {
    console.warn('Failed to start transaction:', error);
  }
  
  // Fallback: return a mock transaction
  return {
    finish: () => {},
    setData: () => {},
    setTag: () => {},
  };
}

