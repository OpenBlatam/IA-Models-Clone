import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Slider as RNSlider } from '@react-native-community/slider';
import { useTheme } from '../context/ThemeContext';

interface SliderProps {
  value: number;
  onValueChange: (value: number) => void;
  minimumValue?: number;
  maximumValue?: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  disabled?: boolean;
}

const Slider: React.FC<SliderProps> = ({
  value,
  onValueChange,
  minimumValue = 0,
  maximumValue = 100,
  step = 1,
  label,
  showValue = true,
  disabled = false,
}) => {
  const { colors } = useTheme();

  return (
    <View style={styles.container}>
      {(label || showValue) && (
        <View style={styles.header}>
          {label && (
            <Text style={[styles.label, { color: colors.text }]}>
              {label}
            </Text>
          )}
          {showValue && (
            <Text style={[styles.value, { color: colors.primary }]}>
              {Math.round(value)}
            </Text>
          )}
        </View>
      )}
      <RNSlider
        style={styles.slider}
        value={value}
        onValueChange={onValueChange}
        minimumValue={minimumValue}
        maximumValue={maximumValue}
        step={step}
        minimumTrackTintColor={colors.primary}
        maximumTrackTintColor={colors.border}
        thumbTintColor={colors.primary}
        disabled={disabled}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
  },
  value: {
    fontSize: 16,
    fontWeight: '600',
  },
  slider: {
    width: '100%',
    height: 40,
  },
});

export default Slider;

