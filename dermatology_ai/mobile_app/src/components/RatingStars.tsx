import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface RatingStarsProps {
  rating: number;
  maxRating?: number;
  size?: number;
  onRatingChange?: (rating: number) => void;
  readonly?: boolean;
}

const RatingStars: React.FC<RatingStarsProps> = ({
  rating,
  maxRating = 5,
  size = 24,
  onRatingChange,
  readonly = false,
}) => {
  const handlePress = (value: number) => {
    if (!readonly && onRatingChange) {
      onRatingChange(value);
    }
  };

  return (
    <View style={styles.container}>
      {Array.from({ length: maxRating }, (_, index) => {
        const value = index + 1;
        const isFilled = value <= Math.round(rating);
        const isHalf = value - 0.5 <= rating && rating < value;

        return (
          <TouchableOpacity
            key={index}
            onPress={() => handlePress(value)}
            disabled={readonly}
            activeOpacity={readonly ? 1 : 0.7}
          >
            <Ionicons
              name={
                isFilled
                  ? 'star'
                  : isHalf
                  ? 'star-half'
                  : 'star-outline'
              }
              size={size}
              color={isFilled || isHalf ? '#fbbf24' : '#d1d5db'}
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
});

export default RatingStars;

