import React, { useMemo } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface CardProps {
  children: React.ReactNode;
  style?: ViewStyle;
  elevated?: boolean;
  padding?: number;
}

const Card: React.FC<CardProps> = ({
  children,
  style,
  elevated = true,
  padding = 16,
}) => {
  const { colors } = useTheme();

  const cardStyle = useMemo(
    () => [
      styles.card,
      {
        backgroundColor: colors.card,
        padding,
        shadowColor: colors.shadow,
      },
      elevated && styles.elevated,
      style,
    ],
    [colors.card, colors.shadow, padding, elevated, style]
  );

  return <View style={cardStyle}>{children}</View>;
};

const styles = StyleSheet.create({
  card: {
    borderRadius: 12,
    marginVertical: 8,
  },
  elevated: {
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
});

export default React.memo(Card);

