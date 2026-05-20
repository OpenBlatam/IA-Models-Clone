import React from 'react';
import { SectionList, SectionListProps, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';

interface StickySectionListProps<T> extends SectionListProps<T> {
  stickySectionHeadersEnabled?: boolean;
}

export function StickySectionList<T>({
  stickySectionHeadersEnabled = true,
  ...props
}: StickySectionListProps<T>) {
  const { theme } = useTheme();

  return (
    <SectionList
      stickySectionHeadersEnabled={stickySectionHeadersEnabled}
      style={[styles.container, { backgroundColor: theme.background }]}
      {...props}
    />
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

