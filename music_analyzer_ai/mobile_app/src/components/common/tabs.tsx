import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet, ScrollView } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface Tab {
  key: string;
  label: string;
  badge?: number;
  disabled?: boolean;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (key: string) => void;
  scrollable?: boolean;
}

/**
 * Tabs component
 * Horizontal tab navigation
 */
export function Tabs({
  tabs,
  activeTab,
  onTabChange,
  scrollable = false,
}: TabsProps) {
  const content = (
    <View style={styles.container}>
      {tabs.map((tab) => {
        const isActive = tab.key === activeTab;
        const isDisabled = tab.disabled;

        return (
          <TouchableOpacity
            key={tab.key}
            style={[
              styles.tab,
              isActive && styles.activeTab,
              isDisabled && styles.disabledTab,
            ]}
            onPress={() => !isDisabled && onTabChange(tab.key)}
            disabled={isDisabled}
            accessibilityRole="tab"
            accessibilityState={{ selected: isActive, disabled: isDisabled }}
          >
            <Text
              style={[
                styles.tabLabel,
                isActive && styles.activeTabLabel,
                isDisabled && styles.disabledTabLabel,
              ]}
            >
              {tab.label}
            </Text>
            {tab.badge !== undefined && tab.badge > 0 && (
              <View style={styles.badge}>
                <Text style={styles.badgeText}>{tab.badge}</Text>
              </View>
            )}
          </TouchableOpacity>
        );
      })}
    </View>
  );

  if (scrollable) {
    return (
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {content}
      </ScrollView>
    );
  }

  return content;
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  scrollContent: {
    flexDirection: 'row',
  },
  tab: {
    paddingVertical: SPACING.md,
    paddingHorizontal: SPACING.lg,
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.xs,
  },
  activeTab: {
    borderBottomColor: COLORS.primary,
  },
  disabledTab: {
    opacity: 0.5,
  },
  tabLabel: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
    fontWeight: '500',
  },
  activeTabLabel: {
    color: COLORS.primary,
    fontWeight: '600',
  },
  disabledTabLabel: {
    color: COLORS.textSecondary,
  },
  badge: {
    backgroundColor: COLORS.error,
    borderRadius: BORDER_RADIUS.full,
    minWidth: 20,
    height: 20,
    paddingHorizontal: SPACING.xs,
    justifyContent: 'center',
    alignItems: 'center',
  },
  badgeText: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.surface,
    fontSize: 10,
    fontWeight: '600',
  },
});

