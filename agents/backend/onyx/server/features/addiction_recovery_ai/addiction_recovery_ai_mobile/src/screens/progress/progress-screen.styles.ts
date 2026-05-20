import { StyleSheet, ViewStyle, TextStyle } from 'react-native';
import type { ColorScheme } from '@/theme/colors';

export function useProgressStyles(colors: ColorScheme) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
    },
    scrollView: {
      flex: 1,
    },
    header: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: 16,
      backgroundColor: colors.surface,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    title: {
      fontSize: 24,
      fontWeight: 'bold',
      color: colors.text,
    },
    logButton: {
      paddingHorizontal: 16,
      paddingVertical: 8,
    },
    logForm: {
      backgroundColor: colors.surface,
      margin: 16,
      padding: 16,
      borderRadius: 12,
      shadowColor: colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    formTitle: {
      fontSize: 18,
      fontWeight: 'bold',
      color: colors.text,
      marginBottom: 16,
    },
    pickerContainer: {
      marginBottom: 16,
    },
    label: {
      fontSize: 14,
      fontWeight: '600',
      color: colors.text,
      marginBottom: 8,
    },
    pickerWrapper: {
      borderWidth: 1,
      borderColor: colors.border,
      borderRadius: 8,
      overflow: 'hidden',
    },
    picker: {
      backgroundColor: colors.surface,
    },
    input: {
      marginBottom: 16,
    },
    saveButton: {
      marginTop: 8,
    },
    cardsContainer: {
      padding: 16,
    },
    statsSection: {
      backgroundColor: colors.surface,
      margin: 16,
      padding: 16,
      borderRadius: 12,
      shadowColor: colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    sectionTitle: {
      fontSize: 20,
      fontWeight: 'bold',
      color: colors.text,
      marginBottom: 16,
    },
    statRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      paddingVertical: 8,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    statLabel: {
      fontSize: 16,
      color: colors.textSecondary,
    },
    statValue: {
      fontSize: 16,
      fontWeight: '600',
      color: colors.text,
    },
    triggersSection: {
      marginTop: 16,
    },
    triggerItem: {
      fontSize: 14,
      color: colors.textSecondary,
      marginTop: 4,
    },
  });
}

