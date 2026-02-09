import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { useInterval } from '../hooks/useInterval';

interface CountdownProps {
  targetDate: Date | string;
  onComplete?: () => void;
  showDays?: boolean;
  showHours?: boolean;
  showMinutes?: boolean;
  showSeconds?: boolean;
  format?: 'short' | 'long';
}

const Countdown: React.FC<CountdownProps> = ({
  targetDate,
  onComplete,
  showDays = true,
  showHours = true,
  showMinutes = true,
  showSeconds = true,
  format = 'short',
}) => {
  const { colors } = useTheme();
  const [timeLeft, setTimeLeft] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0,
  });
  const [isComplete, setIsComplete] = useState(false);

  const calculateTimeLeft = () => {
    const target = typeof targetDate === 'string' ? new Date(targetDate) : targetDate;
    const now = new Date();
    const difference = target.getTime() - now.getTime();

    if (difference <= 0) {
      setIsComplete(true);
      onComplete?.();
      return { days: 0, hours: 0, minutes: 0, seconds: 0 };
    }

    return {
      days: Math.floor(difference / (1000 * 60 * 60 * 24)),
      hours: Math.floor((difference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)),
      minutes: Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60)),
      seconds: Math.floor((difference % (1000 * 60)) / 1000),
    };
  };

  useInterval(() => {
    if (!isComplete) {
      setTimeLeft(calculateTimeLeft());
    }
  }, 1000);

  useEffect(() => {
    setTimeLeft(calculateTimeLeft());
  }, [targetDate]);

  if (isComplete) {
    return (
      <View style={styles.container}>
        <Text style={[styles.completeText, { color: colors.text }]}>
          Completado
        </Text>
      </View>
    );
  }

  const renderTimeUnit = (value: number, label: string, show: boolean) => {
    if (!show) return null;

    return (
      <View style={styles.timeUnit}>
        <Text style={[styles.value, { color: colors.text }]}>
          {String(value).padStart(2, '0')}
        </Text>
        <Text style={[styles.label, { color: colors.textSecondary }]}>
          {format === 'short' ? label[0] : label}
        </Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.timeContainer}>
        {renderTimeUnit(timeLeft.days, 'Días', showDays)}
        {renderTimeUnit(timeLeft.hours, 'Horas', showHours)}
        {renderTimeUnit(timeLeft.minutes, 'Minutos', showMinutes)}
        {renderTimeUnit(timeLeft.seconds, 'Segundos', showSeconds)}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
  },
  timeContainer: {
    flexDirection: 'row',
    gap: 16,
  },
  timeUnit: {
    alignItems: 'center',
    minWidth: 50,
  },
  value: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  label: {
    fontSize: 12,
    marginTop: 4,
  },
  completeText: {
    fontSize: 18,
    fontWeight: '600',
  },
});

export default Countdown;

