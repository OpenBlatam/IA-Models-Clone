import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface Segment {
  label: string;
  value: string;
}

interface SegmentedControlProps {
  segments: Segment[];
  selectedValue: string;
  onValueChange: (value: string) => void;
  size?: 'small' | 'medium' | 'large';
}

const SegmentedControl: React.FC<SegmentedControlProps> = ({
  segments,
  selectedValue,
  onValueChange,
  size = 'medium',
}) => {
  const { colors } = useTheme();
  const [layout, setLayout] = React.useState<Record<string, { x: number; width: number }>>({});
  const indicatorPosition = useSharedValue(0);
  const indicatorWidth = useSharedValue(0);

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { paddingVertical: 6, paddingHorizontal: 12, fontSize: 12 };
      case 'large':
        return { paddingVertical: 12, paddingHorizontal: 20, fontSize: 16 };
      default:
        return { paddingVertical: 8, paddingHorizontal: 16, fontSize: 14 };
    }
  };

  const sizeStyles = getSizeStyles();

  React.useEffect(() => {
    const selectedLayout = layout[selectedValue];
    if (selectedLayout) {
      indicatorPosition.value = withSpring(selectedLayout.x);
      indicatorWidth.value = withSpring(selectedLayout.width);
    }
  }, [selectedValue, layout, indicatorPosition, indicatorWidth]);

  const indicatorStyle = useAnimatedStyle(() => {
    return {
      transform: [{ translateX: indicatorPosition.value }],
      width: indicatorWidth.value,
    };
  });

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: colors.surface,
          borderRadius: 12,
          padding: 4,
        },
      ]}
    >
      {segments.map((segment, index) => {
        const isSelected = selectedValue === segment.value;

        return (
          <TouchableOpacity
            key={segment.value}
            onPress={() => onValueChange(segment.value)}
            onLayout={(e) => {
              const { x, width } = e.nativeEvent.layout;
              setLayout((prev) => ({ ...prev, [segment.value]: { x, width } }));
            }}
            style={[
              styles.segment,
              sizeStyles,
              {
                flex: 1,
              },
            ]}
            activeOpacity={0.7}
          >
            <Text
              style={[
                styles.segmentText,
                {
                  color: isSelected ? colors.primary : colors.textSecondary,
                  fontSize: sizeStyles.fontSize,
                  fontWeight: isSelected ? '600' : '400',
                },
              ]}
            >
              {segment.label}
            </Text>
          </TouchableOpacity>
        );
      })}
      <Animated.View
        style={[
          styles.indicator,
          {
            backgroundColor: colors.primary + '20',
          },
          indicatorStyle,
        ]}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    position: 'relative',
  },
  segment: {
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
  },
  segmentText: {
    textAlign: 'center',
  },
  indicator: {
    position: 'absolute',
    top: 4,
    bottom: 4,
    borderRadius: 8,
    zIndex: 0,
  },
});

export default SegmentedControl;

