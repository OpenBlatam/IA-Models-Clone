import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface DividerProps {
  text?: string;
  orientation?: 'horizontal' | 'vertical';
  thickness?: number;
  spacing?: number;
}

const Divider: React.FC<DividerProps> = ({
  text,
  orientation = 'horizontal',
  thickness = 1,
  spacing = 16,
}) => {
  const { colors } = useTheme();

  if (orientation === 'vertical') {
    return (
      <View
        style={[
          styles.vertical,
          {
            width: thickness,
            backgroundColor: colors.border,
            marginHorizontal: spacing,
          },
        ]}
      />
    );
  }

  if (text) {
    return (
      <View style={styles.horizontalWithText}>
        <View
          style={[
            styles.line,
            {
              flex: 1,
              height: thickness,
              backgroundColor: colors.border,
            },
          ]}
        />
        <Text
          style={[
            styles.text,
            {
              color: colors.textSecondary,
              marginHorizontal: spacing,
            },
          ]}
        >
          {text}
        </Text>
        <View
          style={[
            styles.line,
            {
              flex: 1,
              height: thickness,
              backgroundColor: colors.border,
            },
          ]}
        />
      </View>
    );
  }

  return (
    <View
      style={[
        styles.horizontal,
        {
          height: thickness,
          backgroundColor: colors.border,
          marginVertical: spacing,
        },
      ]}
    />
  );
};

const styles = StyleSheet.create({
  horizontal: {
    width: '100%',
  },
  horizontalWithText: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 16,
  },
  line: {
    height: 1,
  },
  text: {
    fontSize: 14,
  },
  vertical: {
    height: '100%',
  },
});

export default Divider;

