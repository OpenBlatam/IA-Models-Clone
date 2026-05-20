'use client';

import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

interface CountdownProps {
  targetDate: string | Date;
  onComplete?: () => void;
  className?: string;
  showDays?: boolean;
  showHours?: boolean;
  showMinutes?: boolean;
  showSeconds?: boolean;
}

const Countdown = ({
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
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    const target = typeof targetDate === 'string' ? new Date(targetDate) : targetDate;
    const now = new Date();

    if (target <= now) {
      setIsComplete(true);
      if (onComplete) {
        onComplete();
      }
      return;
    }

    const updateCountdown = () => {
      const now = new Date();
      const difference = target.getTime() - now.getTime();

      if (difference <= 0) {
        setIsComplete(true);
        if (onComplete) {
          onComplete();
        }
        return;
      }

      setTimeLeft({
        days: Math.floor(difference / (1000 * 60 * 60 * 24)),
        hours: Math.floor((difference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)),
        minutes: Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60)),
        seconds: Math.floor((difference % (1000 * 60)) / 1000),
      });
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);

    return () => clearInterval(interval);
  }, [targetDate, onComplete]);

  if (isComplete) {
    return <div className={cn('text-gray-500', className)}>Completado</div>;
  }

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {showDays && (
        <div className="text-center">
          <div className="text-2xl font-bold">{String(timeLeft.days).padStart(2, '0')}</div>
          <div className="text-xs text-gray-500">días</div>
        </div>
      )}
      {showHours && (
        <div className="text-center">
          <div className="text-2xl font-bold">{String(timeLeft.hours).padStart(2, '0')}</div>
          <div className="text-xs text-gray-500">horas</div>
        </div>
      )}
      {showMinutes && (
        <div className="text-center">
          <div className="text-2xl font-bold">{String(timeLeft.minutes).padStart(2, '0')}</div>
          <div className="text-xs text-gray-500">min</div>
        </div>
      )}
      {showSeconds && (
        <div className="text-center">
          <div className="text-2xl font-bold">{String(timeLeft.seconds).padStart(2, '0')}</div>
          <div className="text-xs text-gray-500">seg</div>
        </div>
      )}
    </div>
  );
};

export { Countdown };

