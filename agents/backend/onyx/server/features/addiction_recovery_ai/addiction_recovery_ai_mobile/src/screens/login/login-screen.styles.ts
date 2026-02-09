import { StyleSheet, ViewStyle, TextStyle } from 'react-native';
import type { ColorScheme } from '@/theme/colors';

export function useLoginStyles(colors: ColorScheme) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
    },
    keyboardView: {
      flex: 1,
    },
    scrollContent: {
      flexGrow: 1,
      justifyContent: 'center',
      padding: 24,
    },
    content: {
      borderRadius: 16,
      padding: 24,
      backgroundColor: colors.surface,
      shadowColor: colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 8,
      elevation: 4,
    },
    title: {
      fontSize: 32,
      fontWeight: 'bold',
      marginBottom: 8,
      textAlign: 'center',
      color: colors.text,
    },
    subtitle: {
      fontSize: 16,
      marginBottom: 32,
      textAlign: 'center',
      color: colors.textSecondary,
    },
    button: {
      marginTop: 8,
      marginBottom: 16,
    },
    errorText: {
      fontSize: 14,
      marginBottom: 16,
      textAlign: 'center',
      color: colors.error,
    },
    footer: {
      flexDirection: 'row',
      justifyContent: 'center',
      alignItems: 'center',
    },
    footerText: {
      fontSize: 14,
      color: colors.textSecondary,
    },
    linkText: {
      fontSize: 14,
      fontWeight: '600',
      color: colors.primary,
    },
  });
}

