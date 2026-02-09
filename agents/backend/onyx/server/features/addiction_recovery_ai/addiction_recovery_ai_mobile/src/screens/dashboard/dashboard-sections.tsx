import React from 'react';
import { View, Text } from 'react-native';
import { useColors } from '@/theme/colors';
import type { DashboardResponse } from '@/types';
import { useDashboardStyles } from './dashboard-screen.styles';

interface DashboardSectionsProps {
  dashboard: DashboardResponse;
}

export function DashboardSections({ dashboard }: DashboardSectionsProps): JSX.Element {
  const colors = useColors();
  const styles = useDashboardStyles(colors);

  return (
    <>
      {dashboard.achievements && dashboard.achievements.length > 0 && (
        <View style={styles.section}>
          <Text
            style={styles.sectionTitle}
            accessibilityRole="header"
          >
            Logros Recientes
          </Text>
          {dashboard.achievements.slice(0, 3).map((achievement) => (
            <View
              key={achievement.id}
              style={styles.achievementItem}
              accessibilityRole="text"
            >
              <Text style={styles.achievementTitle}>
                {achievement.title}
              </Text>
              <Text style={styles.achievementDescription}>
                {achievement.description}
              </Text>
            </View>
          ))}
        </View>
      )}

      {dashboard.upcoming_reminders && dashboard.upcoming_reminders.length > 0 && (
        <View style={styles.section}>
          <Text
            style={styles.sectionTitle}
            accessibilityRole="header"
          >
            Recordatorios
          </Text>
          {dashboard.upcoming_reminders.slice(0, 3).map((reminder) => (
            <View
              key={reminder.id}
              style={styles.reminderItem}
              accessibilityRole="text"
            >
              <Text style={styles.reminderTitle}>
                {reminder.title}
              </Text>
              <Text style={styles.reminderTime}>
                {reminder.time}
              </Text>
            </View>
          ))}
        </View>
      )}
    </>
  );
}

