'use client';

import { useEffect } from 'react';
import { eventBus } from '@/lib/event-bus';

export function useEventBus(event: string, callback: (...args: any[]) => void) {
  useEffect(() => {
    const unsubscribe = eventBus.on(event, callback);
    return unsubscribe;
  }, [event, callback]);
}

