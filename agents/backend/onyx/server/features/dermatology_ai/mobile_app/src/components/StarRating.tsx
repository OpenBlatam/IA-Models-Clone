import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface StarRatingProps {
  rating: number;
  maxRating?: number;
  size?: number;
  onRatingChange?: (rating: number) => void;
  readonly?: boolean;
  showHalfStars?: boolean;
}

const StarRating: React.FC<StarRatingProps> = ({
  rating,
  maxRating = 5,
  size = 24,
  onRatingChange,
  readonly = false,
  showHalfStars = false,
}) => {
  const { colors } = useTheme();

  const handlePress = (value: number) => {
    if (!readonly && onRatingChange) {
      onRatingChange(value);
    }
  };

  const getStarType = (index: number) => {
    const value = index + 1;
    const halfValue = value - 0.5;

    if (rating >= value) {
      return 'star';
    } else if (showHalfStars && rating >= halfValue) {
      return 'star-half';
    } else {
      return 'star-outline';
    }
  };

  return (
    <View style={styles.container}>
      {Array.from({ length: maxRating }, (_, index) => {
        const value = index + 1;
        const starType = getStarType(index);

        return (
          <TouchableOpacity
            key={index}
            onPress={() => handlePress(value)}
            disabled={readonly}
            activeOpacity={readonly ? 1 : 0.7}
            style={styles.star}
          >
            <Ionicons
              name={starType as any}
              size={size}
              color={rating >= value ? '#fbbf24' : colors.border}
            />
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  star: {
    marginRight: 4,
  },
});

export default StarRating;

