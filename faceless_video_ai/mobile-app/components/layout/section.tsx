import React from 'react';
import { View, Text, ViewProps, StyleSheet } from 'react-native';
import { useTheme } from '@/contexts/theme-context';

interface SectionProps extends ViewProps {
  title?: string;
  children: React.ReactNode;
  headerAction?: React.ReactNode;
}

export function Section({ title, children, headerAction, style, ...props }: SectionProps) {
  const { colors } = useTheme();

  return (
    <View style={[styles.section, style]} {...props}>
      {(title || headerAction) && (
        <View style={styles.header}>
          {title && (
            <Text style={[styles.title, { color: colors.text }]}>{title}</Text>
          )}
          {headerAction && <View>{headerAction}</View>}
        </View>
      )}
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  section: {
    marginBottom: 24,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
  },
});


