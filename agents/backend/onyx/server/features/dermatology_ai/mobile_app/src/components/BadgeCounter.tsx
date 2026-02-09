import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface BadgeCounterProps {
  count: number;
  maxCount?: number;
  size?: 'small' | 'medium' | 'large';
  color?: string;
  showZero?: boolean;
}

const BadgeCounter: React.FC<BadgeCounterProps> = ({
  count,
  maxCount = 99,
  size = 'medium',
  color,
  showZero = false,
}) => {
  const { colors } = useTheme();
  const badgeColor = color || colors.error;

  if (count === 0 && !showZero) return null;

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { fontSize: 10, minWidth: 16, height: 16, paddingHorizontal: 4 };
      case 'large':
        return { fontSize: 14, minWidth: 24, height: 24, paddingHorizontal: 6 };
      default:
        return { fontSize: 12, minWidth: 20, height: 20, paddingHorizontal: 5 };
    }
  };

  const sizeStyles = getSizeStyles();
  const displayCount = count > maxCount ? `${maxCount}+` : count.toString();

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor: badgeColor,
          minWidth: sizeStyles.minWidth,
          height: sizeStyles.height,
          borderRadius: sizeStyles.height / 2,
          paddingHorizontal: sizeStyles.paddingHorizontal,
        },
      ]}
    >
      <Text
        style={[
          styles.text,
          {
            fontSize: sizeStyles.fontSize,
            color: '#fff',
          },
        ]}
      >
        {displayCount}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  badge: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    fontWeight: 'bold',
    color: '#fff',
  },
});

export default BadgeCounter;

