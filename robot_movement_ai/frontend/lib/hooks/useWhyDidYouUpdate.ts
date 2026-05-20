import { useEffect, useRef } from 'react';

/**
 * Debug hook to see what props changed between renders
 * Only works in development
 */
export function useWhyDidYouUpdate(name: string, props: Record<string, any>) {
  if (process.env.NODE_ENV !== 'development') {
    return;
  }

  const previousProps = useRef<Record<string, any>>();

  useEffect(() => {
    if (previousProps.current) {
      const allKeys = Object.keys({ ...previousProps.current, ...props });
      const changedProps: Record<string, { from: any; to: any }> = {};

      allKeys.forEach((key) => {
        if (previousProps.current![key] !== props[key]) {
          changedProps[key] = {
            from: previousProps.current![key],
            to: props[key],
          };
        }
      });

      if (Object.keys(changedProps).length) {
        console.log('[why-did-you-update]', name, changedProps);
      }
    }

    previousProps.current = props;
  });
}



