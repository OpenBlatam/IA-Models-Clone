import React, { useCallback, useMemo } from 'react';
import { Calendar as RNCalendar, DateData } from 'react-native-calendars';
import { useColors } from '@/theme/colors';
import { SPACING } from '@/theme/spacing';

interface CalendarProps {
  onDayPress?: (day: DateData) => void;
  markedDates?: Record<string, any>;
  minDate?: string;
  maxDate?: string;
  current?: string;
  onMonthChange?: (month: DateData) => void;
}

export function Calendar({
  onDayPress,
  markedDates,
  minDate,
  maxDate,
  current,
  onMonthChange,
}: CalendarProps): JSX.Element {
  const colors = useColors();

  const theme = useMemo(
    () => ({
      backgroundColor: colors.surface,
      calendarBackground: colors.surface,
      textSectionTitleColor: colors.textSecondary,
      selectedDayBackgroundColor: colors.primary,
      selectedDayTextColor: '#FFFFFF',
      todayTextColor: colors.primary,
      dayTextColor: colors.text,
      textDisabledColor: colors.textSecondary,
      dotColor: colors.primary,
      selectedDotColor: '#FFFFFF',
      arrowColor: colors.primary,
      monthTextColor: colors.text,
      textDayFontWeight: '600',
      textMonthFontWeight: 'bold',
      textDayHeaderFontWeight: '600',
      textDayFontSize: 16,
      textMonthFontSize: 16,
      textDayHeaderFontSize: 13,
    }),
    [colors]
  );

  const handleDayPress = useCallback(
    (day: DateData) => {
      onDayPress?.(day);
    },
    [onDayPress]
  );

  const handleMonthChange = useCallback(
    (month: DateData) => {
      onMonthChange?.(month);
    },
    [onMonthChange]
  );

  return (
    <RNCalendar
      onDayPress={handleDayPress}
      markedDates={markedDates}
      minDate={minDate}
      maxDate={maxDate}
      current={current}
      onMonthChange={handleMonthChange}
      theme={theme}
      style={{
        borderRadius: 12,
        padding: SPACING.md,
      }}
    />
  );
}

