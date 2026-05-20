export const trackEvent = (eventName: string, properties?: Record<string, unknown>): void => {
  if (typeof window === 'undefined') {
    return;
  }

  // Google Analytics 4
  if ((window as any).gtag) {
    (window as any).gtag('event', eventName, properties);
  }

  // Universal Analytics
  if ((window as any).ga) {
    (window as any).ga('send', 'event', eventName, properties?.action || 'click', properties?.label);
  }

  // Custom analytics
  if ((window as any).analytics && typeof (window as any).analytics.track === 'function') {
    (window as any).analytics.track(eventName, properties);
  }

  // Console log for development
  if (process.env.NODE_ENV === 'development') {
    console.log('Analytics Event:', eventName, properties);
  }
};

export const trackPageView = (path: string, title?: string): void => {
  if (typeof window === 'undefined') {
    return;
  }

  // Google Analytics 4
  if ((window as any).gtag) {
    (window as any).gtag('config', 'GA_MEASUREMENT_ID', {
      page_path: path,
      page_title: title,
    });
  }

  // Universal Analytics
  if ((window as any).ga) {
    (window as any).ga('send', 'pageview', {
      page: path,
      title: title || document.title,
    });
  }

  // Custom analytics
  if ((window as any).analytics && typeof (window as any).analytics.page === 'function') {
    (window as any).analytics.page(path, { title: title || document.title });
  }
};

export const identifyUser = (userId: string, traits?: Record<string, unknown>): void => {
  if (typeof window === 'undefined') {
    return;
  }

  // Google Analytics
  if ((window as any).gtag) {
    (window as any).gtag('set', { user_id: userId });
  }

  // Custom analytics
  if ((window as any).analytics && typeof (window as any).analytics.identify === 'function') {
    (window as any).analytics.identify(userId, traits);
  }
};



