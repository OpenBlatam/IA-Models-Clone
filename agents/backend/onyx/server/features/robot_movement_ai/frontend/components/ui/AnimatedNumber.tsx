'use client';

import { useEffect, useState } from 'react';
import { useSpring, animated } from '@react-spring/web';
import { cn } from '@/lib/utils/cn';

interface AnimatedNumberProps {
  value: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
  format?: (value: number) => string;
}

export default function AnimatedNumber({
  value,
  duration = 1000,
  decimals = 0,
  prefix = '',
  suffix = '',
  className,
  format,
}: AnimatedNumberProps) {
  const [displayValue, setDisplayValue] = useState(value);

  const { number } = useSpring({
    from: { number: displayValue },
    number: value,
    config: { duration },
    onRest: () => setDisplayValue(value),
  });

  const formattedValue = format
    ? format(value)
    : `${prefix}${number.to((n) => n.toFixed(decimals))}${suffix}`;

  return (
    <animated.span className={cn('tabular-nums', className)}>
      {format ? (
        <animated.span>
          {number.to((n) => format(n))}
        </animated.span>
      ) : (
        formattedValue
      )}
    </animated.span>
  );
}



