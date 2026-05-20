export type AnalyticsEvent =
  | { type: 'page_view'; path: string }
  | { type: 'event_created'; eventId: string }
  | { type: 'event_deleted'; eventId: string }
  | { type: 'routine_completed'; routineId: string }
  | { type: 'protocol_viewed'; protocolId: string }
  | { type: 'wardrobe_item_added'; itemId: string }
  | { type: 'search_performed'; query: string }
  | { type: 'button_clicked'; buttonName: string; location: string };

export const trackEvent = (event: AnalyticsEvent) => {
  if (typeof window === 'undefined') return;

  // Aquí puedes integrar con tu servicio de analytics
  // Por ejemplo: Google Analytics, Mixpanel, etc.
  
  if (process.env.NODE_ENV === 'development') {
    console.log('[Analytics]', event);
  }

  // Ejemplo con Google Analytics 4
  // if (window.gtag) {
  //   window.gtag('event', event.type, {
  //     ...event,
  //   });
  // }
};

export const trackPageView = (path: string) => {
  trackEvent({ type: 'page_view', path });
};

