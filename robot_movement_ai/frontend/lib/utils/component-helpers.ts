/**
 * Component helper utilities
 */

import { ReactNode } from 'react';

/**
 * Create a display name for a component
 */
export function getDisplayName(Component: React.ComponentType<any>): string {
  return Component.displayName || Component.name || 'Component';
}

/**
 * Check if component should update (shallow comparison)
 */
export function shouldUpdate(
  prevProps: Record<string, any>,
  nextProps: Record<string, any>,
  keys?: string[]
): boolean {
  const propsToCheck = keys || Object.keys(nextProps);

  for (const key of propsToCheck) {
    if (prevProps[key] !== nextProps[key]) {
      return true;
    }
  }

  return false;
}

/**
 * Merge refs
 */
export function mergeRefs<T = any>(
  ...refs: Array<React.Ref<T> | undefined>
): React.RefCallback<T> {
  return (value: T) => {
    refs.forEach((ref) => {
      if (typeof ref === 'function') {
        ref(value);
      } else if (ref != null) {
        (ref as React.MutableRefObject<T | null>).current = value;
      }
    });
  };
}

/**
 * Get children as array
 */
export function getChildrenAsArray(children: ReactNode): ReactNode[] {
  if (Array.isArray(children)) {
    return children;
  }
  if (children != null) {
    return [children];
  }
  return [];
}

/**
 * Filter children by type
 */
export function filterChildrenByType(
  children: ReactNode,
  type: React.ComponentType<any>
): ReactNode[] {
  return getChildrenAsArray(children).filter(
    (child) =>
      child != null &&
      typeof child === 'object' &&
      'type' in child &&
      child.type === type
  );
}

/**
 * Clone element with merged props
 */
export function cloneElementWithProps(
  element: ReactNode,
  props: Record<string, any>
): ReactNode {
  if (
    element != null &&
    typeof element === 'object' &&
    'type' in element &&
    'props' in element
  ) {
    return {
      ...element,
      props: {
        ...element.props,
        ...props,
      },
    };
  }
  return element;
}

/**
 * Create a forward ref component wrapper
 */
export function forwardRefComponent<T, P = {}>(
  displayName: string,
  component: React.ForwardRefRenderFunction<T, P>
): React.ForwardRefExoticComponent<React.PropsWithoutRef<P> & React.RefAttributes<T>> {
  const ForwardedComponent = React.forwardRef(component);
  ForwardedComponent.displayName = displayName;
  return ForwardedComponent;
}



