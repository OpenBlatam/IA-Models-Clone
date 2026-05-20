import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, SPACING } from '../../constants/config';

interface RatingProps {
  value: number;
  maxValue?: number;
  onValueChange?: (value: number) => void;
  size?: number;
  readonly?: boolean;
  showValue?: boolean;
}

/**
 * Rating component with stars
 */
export function Rating({
  value,
  maxValue = 5,
  onValueChange,
  size = 20,
  readonly = false,
  showValue = false,
}: RatingProps) {
  const handlePress = (index: number) => {
    if (!readonly && onValueChange) {
      onValueChange(index + 1);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.stars}>
        {Array.from({ length: maxValue }).map((_, index) => {
          const isFilled = index < value;
          return (
            <TouchableOpacity
              key={index}
              onPress={() => handlePress(index)}
              disabled={readonly || !onValueChange}
              style={[styles.star, { width: size, height: size }]}
              accessibilityLabel={`${isFilled ? 'Filled' : 'Empty'} star ${index + 1}`}
              accessibilityRole="button"
            >
              <View
                style={[
                  styles.starShape,
                  {
                    width: size,
                    height: size,
                    backgroundColor: isFilled ? COLORS.warning : COLORS.surfaceLight,
                  },
                ]}
              />
            </TouchableOpacity>
          );
        })}
      </View>
      {showValue && (
        <View style={styles.valueContainer}>
          {/* Value display can be added here */}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.xs,
  },
  stars: {
    flexDirection: 'row',
    gap: SPACING.xs,
  },
  star: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  starShape: {
    borderRadius: 4,
  },
  valueContainer: {
    marginLeft: SPACING.sm,
  },
});

