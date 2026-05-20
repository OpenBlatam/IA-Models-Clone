import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { Divider } from './Divider';

interface SectionProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  showDivider?: boolean;
  padding?: boolean;
}

export const Section: React.FC<SectionProps> = ({
  title,
  subtitle,
  children,
  showDivider = false,
  padding = true,
}) => {
  const { theme } = useTheme();

  return (
    <View style={styles.container}>
      {(title || subtitle) && (
        <View style={[styles.header, padding && styles.headerPadding]}>
          {title && (
            <Text style={[styles.title, { color: theme.text }]}>{title}</Text>
          )}
          {subtitle && (
            <Text style={[styles.subtitle, { color: theme.textSecondary }]}>
              {subtitle}
            </Text>
          )}
        </View>
      )}
      <View style={padding && styles.contentPadding}>{children}</View>
      {showDivider && <Divider />}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.lg,
  },
  header: {
    marginBottom: spacing.md,
  },
  headerPadding: {
    paddingHorizontal: spacing.md,
  },
  title: {
    ...typography.h3,
    fontWeight: '600',
    marginBottom: spacing.xs,
  },
  subtitle: {
    ...typography.bodySmall,
  },
  contentPadding: {
    paddingHorizontal: spacing.md,
  },
});

