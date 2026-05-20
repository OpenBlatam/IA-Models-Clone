import React from 'react';
import { View, StyleSheet } from 'react-native';
import Shimmer from './Shimmer';

interface ShimmerTextProps {
  lines?: number;
  width?: number | string;
  lastLineWidth?: number | string;
  height?: number;
}

const ShimmerText: React.FC<ShimmerTextProps> = ({
  lines = 3,
  width = '100%',
  lastLineWidth = '60%',
  height = 16,
}) => {
  return (
    <View style={styles.container}>
      {Array.from({ length: lines }).map((_, index) => (
        <Shimmer
          key={index}
          width={index === lines - 1 ? lastLineWidth : width}
          height={height}
          style={index < lines - 1 ? styles.line : undefined}
        />
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    gap: 8,
  },
  line: {
    marginBottom: 8,
  },
});

export default ShimmerText;

