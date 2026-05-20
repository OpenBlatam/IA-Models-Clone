import React, { useState } from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface RatingProps {
  value: number;
  max?: number;
  onRate?: (value: number) => void;
  readonly?: boolean;
  size?: 'small' | 'medium' | 'large';
  showValue?: boolean;
}

export const Rating: React.FC<RatingProps> = ({
  value,
  max = 5,
  onRate,
  readonly = false,
  size = 'medium',
  showValue = false,
}) => {
  const { theme } = useTheme();
  const [hoveredValue, setHoveredValue] = useState<number | null>(null);

  const getSize = () => {
    switch (size) {
      case 'small':
        return 16;
      case 'large':
        return 32;
      default:
        return 24;
    }
  };

  const starSize = getSize();
  const displayValue = hoveredValue ?? value;

  const handlePress = (newValue: number) => {
    if (!readonly && onRate) {
      hapticFeedback.selection();
      onRate(newValue);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.starsContainer}>
        {Array.from({ length: max }, (_, i) => i + 1).map((starValue) => (
          <TouchableOpacity
            key={starValue}
            onPress={() => handlePress(starValue)}
            onPressIn={() => !readonly && setHoveredValue(starValue)}
            onPressOut={() => setHoveredValue(null)}
            disabled={readonly}
            activeOpacity={readonly ? 1 : 0.7}
          >
            <Text
              style={[
                styles.star,
                {
                  fontSize: starSize,
                  color: starValue <= displayValue ? theme.warning : theme.border,
                },
              ]}
            >
              ★
            </Text>
          </TouchableOpacity>
        ))}
      </View>
      {showValue && (
        <Text style={[styles.value, { color: theme.textSecondary }]}>
          {value.toFixed(1)}/{max}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  starsContainer: {
    flexDirection: 'row',
    gap: spacing.xs,
  },
  star: {
    lineHeight: 24,
  },
  value: {
    ...typography.bodySmall,
    marginLeft: spacing.xs,
  },
});

