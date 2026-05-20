import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  variant?: 'default' | 'pills';
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeTab,
  onTabChange,
  variant = 'default',
}) => {
  const { theme } = useTheme();

  const handleTabPress = (tabId: string) => {
    hapticFeedback.selection();
    onTabChange(tabId);
  };

  if (variant === 'pills') {
    return (
      <View style={styles.pillsContainer}>
        {tabs.map((tab) => (
          <TouchableOpacity
            key={tab.id}
            style={[
              styles.pill,
              {
                backgroundColor: activeTab === tab.id ? theme.primary : theme.surfaceVariant,
              },
            ]}
            onPress={() => handleTabPress(tab.id)}
            activeOpacity={0.7}
          >
            {tab.icon && <View style={styles.iconContainer}>{tab.icon}</View>}
            <Text
              style={[
                styles.pillText,
                {
                  color: activeTab === tab.id ? theme.surface : theme.text,
                },
              ]}
            >
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  }

  return (
    <View style={[styles.container, { backgroundColor: theme.surface, borderBottomColor: theme.border }]}>
      <View style={styles.tabsRow}>
        {tabs.map((tab) => (
          <TouchableOpacity
            key={tab.id}
            style={[
              styles.tab,
              activeTab === tab.id && [
                styles.activeTab,
                { borderBottomColor: theme.primary },
              ],
            ]}
            onPress={() => handleTabPress(tab.id)}
            activeOpacity={0.7}
          >
            {tab.icon && <View style={styles.iconContainer}>{tab.icon}</View>}
            <Text
              style={[
                styles.tabText,
                {
                  color: activeTab === tab.id ? theme.primary : theme.textSecondary,
                },
              ]}
            >
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderBottomWidth: 1,
  },
  tabsRow: {
    flexDirection: 'row',
  },
  tab: {
    flex: 1,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    alignItems: 'center',
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
  },
  activeTab: {
    borderBottomWidth: 2,
  },
  tabText: {
    ...typography.bodySmall,
    fontWeight: '500',
  },
  pillsContainer: {
    flexDirection: 'row',
    gap: spacing.sm,
    padding: spacing.sm,
  },
  pill: {
    flex: 1,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.full,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.xs,
  },
  pillText: {
    ...typography.bodySmall,
    fontWeight: '600',
  },
  iconContainer: {
    marginRight: spacing.xs,
  },
});

