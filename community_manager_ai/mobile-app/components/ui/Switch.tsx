import { TouchableOpacity, View, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';

interface SwitchProps {
  value: boolean;
  onValueChange: (value: boolean) => void;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
  style?: ViewStyle;
  accessibilityLabel?: string;
}

const SIZES = {
  small: { width: 36, height: 20, thumb: 16 },
  medium: { width: 48, height: 28, thumb: 24 },
  large: { width: 60, height: 36, thumb: 32 },
};

export function Switch({
  value,
  onValueChange,
  disabled = false,
  size = 'medium',
  style,
  accessibilityLabel,
}: SwitchProps) {
  const translateX = useSharedValue(value ? 1 : 0);
  const sizeConfig = SIZES[size];

  useEffect(() => {
    translateX.value = withSpring(value ? 1 : 0, {
      damping: 15,
      stiffness: 150,
    });
  }, [value]);

  const animatedStyle = useAnimatedStyle(() => {
    const maxTranslate = sizeConfig.width - sizeConfig.thumb - 4;
    return {
      transform: [{ translateX: translateX.value * maxTranslate }],
    };
  });

  const handlePress = () => {
    if (!disabled) {
      onValueChange(!value);
    }
  };

  return (
    <TouchableOpacity
      onPress={handlePress}
      disabled={disabled}
      activeOpacity={0.7}
      accessibilityRole="switch"
      accessibilityState={{ checked: value, disabled }}
      accessibilityLabel={accessibilityLabel}
      style={style}
    >
      <View
        style={[
          styles.track,
          {
            width: sizeConfig.width,
            height: sizeConfig.height,
            backgroundColor: value ? '#0ea5e9' : '#d1d5db',
            opacity: disabled ? 0.5 : 1,
          },
        ]}
      >
        <Animated.View
          style={[
            styles.thumb,
            {
              width: sizeConfig.thumb,
              height: sizeConfig.thumb,
            },
            animatedStyle,
          ]}
        />
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  track: {
    borderRadius: 20,
    justifyContent: 'center',
    padding: 2,
  },
  thumb: {
    backgroundColor: '#fff',
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
});

