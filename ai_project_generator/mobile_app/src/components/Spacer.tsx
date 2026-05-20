import React from 'react';
import { View, StyleSheet } from 'react-native';
import { spacing } from '../theme/colors';

interface SpacerProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
  horizontal?: boolean;
}

export const Spacer: React.FC<SpacerProps> = ({
  size = 'md',
  horizontal = false,
}) => {
  const getSize = () => {
    switch (size) {
      case 'xs':
        return spacing.xs;
      case 'sm':
        return spacing.sm;
      case 'lg':
        return spacing.lg;
      case 'xl':
        return spacing.xl;
      case 'xxl':
        return spacing.xxl;
      default:
        return spacing.md;
    }
  };

  return (
    <View
      style={[
        horizontal ? styles.horizontal : styles.vertical,
        {
          [horizontal ? 'width' : 'height']: getSize(),
        },
      ]}
    />
  );
};

const styles = StyleSheet.create({
  vertical: {
    width: '100%',
  },
  horizontal: {
    height: '100%',
  },
});

