export const captureException = (error: Error, context?: Record<string, unknown>): void => {
  if (typeof window === 'undefined') {
    return;
  }

  // Sentry
  if ((window as any).Sentry) {
    (window as any).Sentry.captureException(error, { contexts: { custom: context } });
  }

  // LogRocket
  if ((window as any).LogRocket) {
    (window as any).LogRocket.captureException(error);
  }

  // Custom error tracking
  if ((window as any).errorTracker && typeof (window as any).errorTracker.capture === 'function') {
    (window as any).errorTracker.capture(error, context);
  }

  // Console for development
  if (process.env.NODE_ENV === 'development') {
    console.error('Exception captured:', error, context);
  }
};

export const captureMessage = (message: string, level: 'info' | 'warning' | 'error' = 'info'): void => {
  if (typeof window === 'undefined') {
    return;
  }

  // Sentry
  if ((window as any).Sentry) {
    (window as any).Sentry.captureMessage(message, level);
  }

  // LogRocket
  if ((window as any).LogRocket) {
    (window as any).LogRocket.captureMessage(message);
  }

  // Console for development
  if (process.env.NODE_ENV === 'development') {
    console[level](message);
  }
};

export const setUserContext = (userId: string, userData?: Record<string, unknown>): void => {
  if (typeof window === 'undefined') {
    return;
  }

  // Sentry
  if ((window as any).Sentry) {
    (window as any).Sentry.setUser({ id: userId, ...userData });
  }

  // LogRocket
  if ((window as any).LogRocket) {
    (window as any).LogRocket.identify(userId, userData);
  }
};



