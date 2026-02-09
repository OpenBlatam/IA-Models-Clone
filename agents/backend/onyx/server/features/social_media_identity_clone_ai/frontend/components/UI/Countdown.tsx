import { memo, useEffect, useState } from 'react';
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

const Countdown = memo(({
  targetDate,
  onComplete,
  className = '',
  showDays = true,
  showHours = true,
  showMinutes = true,
  showSeconds = true,
}: CountdownProps): JSX.Element => {
  const [timeLeft, setTimeLeft] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0,
  });

  useEffect(() => {
    const target = new Date(targetDate).getTime();

    const updateCountdown = (): void => {
      const now = Date.now();
      const difference = target - now;

      if (difference <= 0) {
        setTimeLeft({ days: 0, hours: 0, minutes: 0, seconds: 0 });
        if (onComplete) {
          onComplete();
        }
        return;
      }

      const days = Math.floor(difference / (1000 * 60 * 60 * 24));
      const hours = Math.floor((difference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((difference % (1000 * 60)) / 1000);

      setTimeLeft({ days, hours, minutes, seconds });
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);

    return () => clearInterval(interval);
  }, [targetDate, onComplete]);

  const parts: Array<{ label: string; value: number }> = [];

  if (showDays) parts.push({ label: 'Days', value: timeLeft.days });
  if (showHours) parts.push({ label: 'Hours', value: timeLeft.hours });
  if (showMinutes) parts.push({ label: 'Minutes', value: timeLeft.minutes });
  if (showSeconds) parts.push({ label: 'Seconds', value: timeLeft.seconds });

  return (
    <div className={cn('flex gap-4', className)}>
      {parts.map((part) => (
        <div key={part.label} className="text-center">
          <div className="text-2xl font-bold tabular-nums">{String(part.value).padStart(2, '0')}</div>
          <div className="text-xs text-gray-500 uppercase">{part.label}</div>
        </div>
      ))}
    </div>
  );
});

Countdown.displayName = 'Countdown';

export default Countdown;



