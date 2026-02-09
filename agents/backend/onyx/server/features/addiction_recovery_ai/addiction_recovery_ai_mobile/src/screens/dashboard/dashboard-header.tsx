import React from 'react';
import { View, Text } from 'react-native';
import { format } from 'date-fns';
import { useColors } from '@/theme/colors';
import type { UserResponse } from '@/types';
import { useDashboardStyles } from './dashboard-screen.styles';

interface DashboardHeaderProps {
  user: UserResponse | null;
}

export function DashboardHeader({ user }: DashboardHeaderProps): JSX.Element {
  const colors = useColors();
  const styles = useDashboardStyles(colors);

  return (
    <View style={styles.header}>
      <Text
        style={styles.greeting}
        accessibilityRole="text"
      >
        Hola, {user?.name || user?.user_id}
      </Text>
      <Text
        style={styles.date}
        accessibilityRole="text"
      >
        {format(new Date(), "EEEE, d 'de' MMMM")}
      </Text>
    </View>
  );
}

