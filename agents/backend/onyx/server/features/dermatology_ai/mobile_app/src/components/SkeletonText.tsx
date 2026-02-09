import React from 'react';
import { View, StyleSheet } from 'react-native';
import Skeleton from './Skeleton';

interface SkeletonTextProps {
  lines?: number;
  width?: number | string;
  lastLineWidth?: number | string;
}

const SkeletonText: React.FC<SkeletonTextProps> = ({
  lines = 3,
  width = '100%',
  lastLineWidth = '60%',
}) => {
  return (
    <View style={styles.container}>
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton
          key={index}
          width={index === lines - 1 ? lastLineWidth : width}
          height={16}
          variant="text"
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

export default SkeletonText;

