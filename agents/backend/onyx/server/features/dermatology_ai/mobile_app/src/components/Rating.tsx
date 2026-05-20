import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface RatingProps {
  rating: number;
  onRatingChange?: (rating: number) => void;
  maxRating?: number;
  size?: number;
  readonly?: boolean;
  showLabel?: boolean;
}

const Rating: React.FC<RatingProps> = ({
  rating,
  onRatingChange,
  maxRating = 5,
  size = 24,
  readonly = false,
  showLabel = false,
}) => {
  const { colors } = useTheme();
  const [currentRating, setCurrentRating] = React.useState(rating);

  const handlePress = (value: number) => {
    if (!readonly && onRatingChange) {
      setCurrentRating(value);
      onRatingChange(value);
    }
  };

  const displayRating = readonly ? rating : currentRating;

  return (
    <View style={styles.container}>
      <View style={styles.stars}>
        {Array.from({ length: maxRating }, (_, index) => {
          const value = index + 1;
          const isFilled = value <= displayRating;
          const scale = useSharedValue(1);

          const animatedStyle = useAnimatedStyle(() => {
            return {
              transform: [{ scale: scale.value }],
            };
          });

          const handlePressIn = () => {
            if (!readonly) {
              scale.value = withSpring(0.9);
            }
          };

          const handlePressOut = () => {
            if (!readonly) {
              scale.value = withSpring(1);
            }
          };

          return (
            <TouchableOpacity
              key={index}
              onPress={() => handlePress(value)}
              onPressIn={handlePressIn}
              onPressOut={handlePressOut}
              disabled={readonly}
              activeOpacity={0.7}
            >
              <Animated.View style={animatedStyle}>
                <Ionicons
                  name={isFilled ? 'star' : 'star-outline'}
                  size={size}
                  color={isFilled ? colors.warning || '#fbbf24' : colors.border}
                />
              </Animated.View>
            </TouchableOpacity>
          );
        })}
      </View>
      {showLabel && (
        <View style={styles.label}>
          <Text style={[styles.labelText, { color: colors.text }]}>
            {displayRating.toFixed(1)} / {maxRating}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  stars: {
    flexDirection: 'row',
    gap: 4,
  },
  label: {
    marginLeft: 8,
  },
  labelText: {
    fontSize: 14,
    fontWeight: '500',
  },
});

export default Rating;

