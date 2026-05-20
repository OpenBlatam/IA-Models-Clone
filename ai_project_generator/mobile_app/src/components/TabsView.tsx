import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
  badge?: number;
}

interface TabsViewProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  variant?: 'default' | 'pills' | 'underline';
}

export const TabsView: React.FC<TabsViewProps> = ({
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

  const getVariantStyles = (isActive: boolean) => {
    switch (variant) {
      case 'pills':
        return {
          container: {
            backgroundColor: isActive ? theme.primary : theme.surfaceVariant,
            borderRadius: borderRadius.full,
          },
          text: {
            color: isActive ? theme.surface : theme.text,
          },
        };
      case 'underline':
        return {
          container: {
            borderBottomWidth: isActive ? 2 : 0,
            borderBottomColor: theme.primary,
          },
          text: {
            color: isActive ? theme.primary : theme.textSecondary,
          },
        };
      default:
        return {
          container: {
            backgroundColor: isActive ? theme.surfaceVariant : 'transparent',
            borderRadius: borderRadius.md,
          },
          text: {
            color: isActive ? theme.primary : theme.text,
            fontWeight: isActive ? '600' : '400',
          },
        };
    }
  };

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: theme.surface,
          borderBottomColor: theme.border,
        },
      ]}
    >
      {tabs.map((tab) => {
        const isActive = tab.id === activeTab;
        const variantStyles = getVariantStyles(isActive);

        return (
          <TouchableOpacity
            key={tab.id}
            style={[
              styles.tab,
              variantStyles.container,
            ]}
            onPress={() => handleTabPress(tab.id)}
            activeOpacity={0.7}
          >
            {tab.icon && <View style={styles.icon}>{tab.icon}</View>}
            <Text style={[styles.label, variantStyles.text]}>
              {tab.label}
            </Text>
            {tab.badge !== undefined && tab.badge > 0 && (
              <View
                style={[
                  styles.badge,
                  {
                    backgroundColor: theme.error,
                  },
                ]}
              >
                <Text style={[styles.badgeText, { color: theme.surface }]}>
                  {tab.badge > 99 ? '99+' : tab.badge}
                </Text>
              </View>
            )}
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
  },
  tab: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    marginRight: spacing.sm,
  },
  icon: {
    marginRight: spacing.xs,
  },
  label: {
    ...typography.bodySmall,
  },
  badge: {
    minWidth: 18,
    height: 18,
    borderRadius: 9,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: spacing.xs,
    marginLeft: spacing.xs,
  },
  badgeText: {
    ...typography.caption,
    fontSize: 10,
    fontWeight: '600',
  },
});

