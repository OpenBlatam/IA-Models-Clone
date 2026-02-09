import { StyleSheet, ViewStyle, TextStyle } from 'react-native';
import type { ColorScheme } from '@/theme/colors';

export function useDashboardStyles(colors: ColorScheme) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
    },
    scrollView: {
      flex: 1,
    },
    scrollContent: {
      paddingBottom: 16,
    },
    emptyContainer: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
    },
    emptyText: {
      fontSize: 16,
      color: colors.textSecondary,
    },
    header: {
      padding: 24,
      backgroundColor: colors.surface,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    greeting: {
      fontSize: 28,
      fontWeight: 'bold',
      marginBottom: 4,
      color: colors.text,
    },
    date: {
      fontSize: 16,
      color: colors.textSecondary,
    },
    cardsContainer: {
      padding: 16,
    },
    riskCard: {
      borderRadius: 12,
      padding: 16,
      marginBottom: 12,
      backgroundColor: colors.card,
      shadowColor: colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    riskTitle: {
      fontSize: 14,
      marginBottom: 4,
      color: colors.textSecondary,
    },
    riskValue: {
      fontSize: 24,
      fontWeight: 'bold',
    },
    section: {
      padding: 16,
      marginTop: 8,
      marginHorizontal: 16,
      borderRadius: 12,
      marginBottom: 16,
      backgroundColor: colors.surface,
    },
    sectionTitle: {
      fontSize: 20,
      fontWeight: 'bold',
      marginBottom: 12,
      color: colors.text,
    },
    achievementItem: {
      paddingVertical: 12,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    achievementTitle: {
      fontSize: 16,
      fontWeight: '600',
      marginBottom: 4,
      color: colors.text,
    },
    achievementDescription: {
      fontSize: 14,
      color: colors.textSecondary,
    },
    reminderItem: {
      paddingVertical: 12,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    reminderTitle: {
      fontSize: 16,
      fontWeight: '600',
      marginBottom: 4,
      color: colors.text,
    },
    reminderTime: {
      fontSize: 14,
      color: colors.textSecondary,
    },
  });
}

