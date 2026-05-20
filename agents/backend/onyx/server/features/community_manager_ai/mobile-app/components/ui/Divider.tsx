import { View, StyleSheet, ViewStyle } from 'react-native';

interface DividerProps {
  orientation?: 'horizontal' | 'vertical';
  spacing?: number;
  style?: ViewStyle;
}

export function Divider({ orientation = 'horizontal', spacing = 16, style }: DividerProps) {
  return (
    <View
      style={[
        styles.divider,
        orientation === 'horizontal' ? styles.horizontal : styles.vertical,
        { margin: spacing },
        style,
      ]}
    />
  );
}

const styles = StyleSheet.create({
  divider: {
    backgroundColor: '#e5e7eb',
  },
  horizontal: {
    height: 1,
    width: '100%',
  },
  vertical: {
    width: 1,
    height: '100%',
  },
});


