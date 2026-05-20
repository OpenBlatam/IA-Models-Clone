import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface Segment {
  label: string;
  value: string;
  icon?: React.ReactNode;
}

interface SegmentedControlProps {
  segments: Segment[];
  selectedValue: string;
  onValueChange: (value: string) => void;
  size?: 'small' | 'medium' | 'large';
}

export const SegmentedControl: React.FC<SegmentedControlProps> = ({
  segments,
  selectedValue,
  onValueChange,
  size = 'medium',
}) => {
  const { theme } = useTheme();

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { padding: spacing.xs, fontSize: typography.caption.fontSize };
      case 'large':
        return { padding: spacing.md, fontSize: typography.body.fontSize };
      default:
        return { padding: spacing.sm, fontSize: typography.bodySmall.fontSize };
    }
  };

  const sizeStyles = getSizeStyles();

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: theme.surfaceVariant,
          borderRadius: borderRadius.lg,
          padding: 2,
        },
      ]}
    >
      <View style={styles.segments}>
        {segments.map((segment, index) => {
          const isSelected = segment.value === selectedValue;
          return (
            <TouchableOpacity
              key={segment.value}
              style={[
                styles.segment,
                {
                  backgroundColor: isSelected ? theme.surface : 'transparent',
                  paddingVertical: sizeStyles.padding,
                  paddingHorizontal: spacing.md,
                },
                index === 0 && styles.firstSegment,
                index === segments.length - 1 && styles.lastSegment,
              ]}
              onPress={() => {
                hapticFeedback.selection();
                onValueChange(segment.value);
              }}
              activeOpacity={0.7}
            >
              {segment.icon && (
                <View style={styles.iconContainer}>{segment.icon}</View>
              )}
              <Text
                style={[
                  styles.segmentText,
                  {
                    fontSize: sizeStyles.fontSize,
                    color: isSelected ? theme.primary : theme.textSecondary,
                    fontWeight: isSelected ? '600' : '400',
                  },
                ]}
              >
                {segment.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
  },
  segments: {
    flexDirection: 'row',
    flex: 1,
  },
  segment: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: borderRadius.md,
    gap: spacing.xs,
  },
  firstSegment: {
    borderTopLeftRadius: borderRadius.md,
    borderBottomLeftRadius: borderRadius.md,
  },
  lastSegment: {
    borderTopRightRadius: borderRadius.md,
    borderBottomRightRadius: borderRadius.md,
  },
  iconContainer: {
    marginRight: spacing.xs,
  },
  segmentText: {
    ...typography.bodySmall,
  },
});

