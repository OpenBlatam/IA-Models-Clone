import { useState, useEffect } from 'react';
import { useInterval } from './useInterval';

interface CountdownState {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
  totalSeconds: number;
  isComplete: boolean;
}

export const useCountdown = (
  targetDate: Date | string,
  onComplete?: () => void
): CountdownState => {
  const [state, setState] = useState<CountdownState>(() => {
    const target = typeof targetDate === 'string' ? new Date(targetDate) : targetDate;
    const now = new Date();
    const difference = target.getTime() - now.getTime();

    if (difference <= 0) {
      return {
        days: 0,
        hours: 0,
        minutes: 0,
        seconds: 0,
        totalSeconds: 0,
        isComplete: true,
      };
    }

    const totalSeconds = Math.floor(difference / 1000);
    return {
      days: Math.floor(totalSeconds / (60 * 60 * 24)),
      hours: Math.floor((totalSeconds % (60 * 60 * 24)) / (60 * 60)),
      minutes: Math.floor((totalSeconds % (60 * 60)) / 60),
      seconds: totalSeconds % 60,
      totalSeconds,
      isComplete: false,
    };
  });

  useInterval(() => {
    if (state.isComplete) return;

    const target = typeof targetDate === 'string' ? new Date(targetDate) : targetDate;
    const now = new Date();
    const difference = target.getTime() - now.getTime();

    if (difference <= 0) {
      setState({
        days: 0,
        hours: 0,
        minutes: 0,
        seconds: 0,
        totalSeconds: 0,
        isComplete: true,
      });
      onComplete?.();
      return;
    }

    const totalSeconds = Math.floor(difference / 1000);
    setState({
      days: Math.floor(totalSeconds / (60 * 60 * 24)),
      hours: Math.floor((totalSeconds % (60 * 60 * 24)) / (60 * 60)),
      minutes: Math.floor((totalSeconds % (60 * 60)) / 60),
      seconds: totalSeconds % 60,
      totalSeconds,
      isComplete: false,
    });
  }, 1000);

  useEffect(() => {
    const target = typeof targetDate === 'string' ? new Date(targetDate) : targetDate;
    const now = new Date();
    const difference = target.getTime() - now.getTime();

    if (difference <= 0) {
      setState({
        days: 0,
        hours: 0,
        minutes: 0,
        seconds: 0,
        totalSeconds: 0,
        isComplete: true,
      });
    }
  }, [targetDate]);

  return state;
};

