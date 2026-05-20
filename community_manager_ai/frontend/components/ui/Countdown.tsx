'use client';

import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface CountdownProps {
  targetDate: Date | string;
  onComplete?: () => void;
  className?: string;
  showDays?: boolean;
  showHours?: boolean;
  showMinutes?: boolean;
  showSeconds?: boolean;
}

export const Countdown = ({
  targetDate,
  onComplete,
  className,
  showDays = true,
  showHours = true,
  showMinutes = true,
  showSeconds = true,
}: CountdownProps) => {
  const [timeLeft, setTimeLeft] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0,
  });

  useEffect(() => {
    const calculateTimeLeft = () => {
      const target = new Date(targetDate).getTime();
      const now = new Date().getTime();
      const difference = target - now;

      if (difference <= 0) {
        setTimeLeft({ days: 0, hours: 0, minutes: 0, seconds: 0 });
        onComplete?.();
        return;
      }

      setTimeLeft({
        days: Math.floor(difference / (1000 * 60 * 60 * 24)),
        hours: Math.floor((difference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)),
        minutes: Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60)),
        seconds: Math.floor((difference % (1000 * 60)) / 1000),
      });
    };

    calculateTimeLeft();
    const interval = setInterval(calculateTimeLeft, 1000);

    return () => clearInterval(interval);
  }, [targetDate, onComplete]);

  const parts = [];

  if (showDays && timeLeft.days > 0) {
    parts.push(`${timeLeft.days}d`);
  }
  if (showHours && (timeLeft.hours > 0 || timeLeft.days > 0)) {
    parts.push(`${timeLeft.hours}h`);
  }
  if (showMinutes && (timeLeft.minutes > 0 || timeLeft.hours > 0 || timeLeft.days > 0)) {
    parts.push(`${timeLeft.minutes}m`);
  }
  if (showSeconds) {
    parts.push(`${timeLeft.seconds}s`);
  }

  if (parts.length === 0) {
    return null;
  }

  return (
    <div className={cn('flex items-center gap-1 text-sm font-medium', className)}>
      {parts.map((part, index) => (
        <span key={index} className="text-gray-900 dark:text-gray-100">
          {part}
        </span>
      ))}
    </div>
  );
};



