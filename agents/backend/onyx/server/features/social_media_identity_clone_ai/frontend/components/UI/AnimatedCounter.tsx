import { memo, useEffect, useState, useRef } from 'react';
import { useIntersectionObserver } from '@/lib/hooks';
import { cn } from '@/lib/utils';

interface AnimatedCounterProps {
  value: number;
  duration?: number;
  className?: string;
  formatter?: (value: number) => string;
}

const AnimatedCounter = memo(({
  value,
  duration = 2000,
  className = '',
  formatter = (val) => val.toLocaleString(),
}: AnimatedCounterProps): JSX.Element => {
  const [displayValue, setDisplayValue] = useState(0);
  const { elementRef, isIntersecting } = useIntersectionObserver<HTMLSpanElement>({
    triggerOnce: true,
  });
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    if (!isIntersecting) {
      return;
    }

    const startTime = Date.now();
    const startValue = displayValue;

    const animate = (): void => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const currentValue = Math.floor(startValue + (value - startValue) * easeOutQuart);

      setDisplayValue(currentValue);

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        setDisplayValue(value);
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [value, duration, isIntersecting, displayValue]);

  return (
    <span ref={elementRef} className={cn('tabular-nums', className)}>
      {formatter(displayValue)}
    </span>
  );
});

AnimatedCounter.displayName = 'AnimatedCounter';

export default AnimatedCounter;



