'use client';

import { memo, useState, useCallback, useRef, useEffect } from 'react';
import React from 'react';
import { createPortal } from 'react-dom';
import { cn } from '@/lib/utils';
import { useEventListener } from '@/lib/hooks';

interface TooltipProps {
  children: React.ReactElement;
  content: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
  className?: string;
  disabled?: boolean;
}

const Tooltip = memo(
  ({
    children,
    content,
    position = 'top',
    delay = 200,
    className,
    disabled = false,
  }: TooltipProps): JSX.Element => {
    const [isVisible, setIsVisible] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
    const timeoutRef = useRef<NodeJS.Timeout>();
    const triggerRef = useRef<HTMLElement>(null);
    const tooltipRef = useRef<HTMLDivElement>(null);

    const calculatePosition = useCallback((): void => {
      if (!triggerRef.current || !tooltipRef.current) return;

      const triggerRect = triggerRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      const scrollY = window.scrollY;
      const scrollX = window.scrollX;

      let top = 0;
      let left = 0;

      switch (position) {
        case 'top':
          top = triggerRect.top + scrollY - tooltipRect.height - 8;
          left = triggerRect.left + scrollX + triggerRect.width / 2 - tooltipRect.width / 2;
          break;
        case 'bottom':
          top = triggerRect.bottom + scrollY + 8;
          left = triggerRect.left + scrollX + triggerRect.width / 2 - tooltipRect.width / 2;
          break;
        case 'left':
          top = triggerRect.top + scrollY + triggerRect.height / 2 - tooltipRect.height / 2;
          left = triggerRect.left + scrollX - tooltipRect.width - 8;
          break;
        case 'right':
          top = triggerRect.top + scrollY + triggerRect.height / 2 - tooltipRect.height / 2;
          left = triggerRect.right + scrollX + 8;
          break;
      }

      setTooltipPosition({ top, left });
    }, [position]);

    const showTooltip = useCallback((): void => {
      if (disabled) return;
      timeoutRef.current = setTimeout(() => {
        setIsVisible(true);
        setTimeout(calculatePosition, 0);
      }, delay);
    }, [disabled, delay, calculatePosition]);

    const hideTooltip = useCallback((): void => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      setIsVisible(false);
    }, []);

    useEventListener('scroll', hideTooltip, window, true);
    useEventListener('resize', calculatePosition, window);

    useEffect(() => {
      if (isVisible) {
        calculatePosition();
      }
    }, [isVisible, calculatePosition]);

    const child = children as React.ReactElement & { ref?: React.Ref<HTMLElement> };
    const trigger = React.cloneElement(child, {
      ref: (node: HTMLElement) => {
        triggerRef.current = node;
        if (typeof child.ref === 'function') {
          child.ref(node);
        } else if (child.ref) {
          (child.ref as React.MutableRefObject<HTMLElement>).current = node;
        }
      },
      onMouseEnter: showTooltip,
      onMouseLeave: hideTooltip,
      onFocus: showTooltip,
      onBlur: hideTooltip,
    });

    if (disabled || !content) {
      return children;
    }

    return (
      <>
        {trigger}
        {isVisible &&
          typeof window !== 'undefined' &&
          createPortal(
            <div
              ref={tooltipRef}
              className={cn(
                'absolute z-50 px-2 py-1 text-xs text-white bg-gray-900 rounded shadow-lg pointer-events-none',
                className
              )}
              style={{
                top: `${tooltipPosition.top}px`,
                left: `${tooltipPosition.left}px`,
              }}
              role="tooltip"
              aria-hidden={!isVisible}
            >
              {content}
            </div>,
            document.body
          )}
      </>
    );
  }
);

Tooltip.displayName = 'Tooltip';

export default Tooltip;
