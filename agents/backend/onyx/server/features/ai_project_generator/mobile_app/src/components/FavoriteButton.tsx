import React, { useState, useEffect } from 'react';
import { TouchableOpacity, Text, StyleSheet, Animated } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius } from '../theme/colors';
import { storage, STORAGE_KEYS } from '../utils/storage';
import { hapticFeedback } from '../utils/haptics';

interface FavoriteButtonProps {
  projectId: string;
  size?: number;
  onToggle?: (isFavorite: boolean) => void;
}

export const FavoriteButton: React.FC<FavoriteButtonProps> = ({
  projectId,
  size = 24,
  onToggle,
}) => {
  const { theme } = useTheme();
  const [isFavorite, setIsFavorite] = useState(false);
  const scale = React.useRef(new Animated.Value(1)).current;

  useEffect(() => {
    loadFavoriteStatus();
  }, [projectId]);

  const loadFavoriteStatus = async () => {
    try {
      const favorites = await storage.get<string[]>(STORAGE_KEYS.FAVORITES) || [];
      setIsFavorite(favorites.includes(projectId));
    } catch (error) {
      console.error('Error loading favorite status:', error);
    }
  };

  const toggleFavorite = async () => {
    try {
      const favorites = await storage.get<string[]>(STORAGE_KEYS.FAVORITES) || [];
      let newFavorites: string[];
      
      if (isFavorite) {
        newFavorites = favorites.filter(id => id !== projectId);
        hapticFeedback.light();
      } else {
        newFavorites = [...favorites, projectId];
        hapticFeedback.success();
      }
      
      await storage.set(STORAGE_KEYS.FAVORITES, newFavorites);
      setIsFavorite(!isFavorite);
      onToggle?.(!isFavorite);

      Animated.sequence([
        Animated.spring(scale, {
          toValue: 1.3,
          useNativeDriver: true,
          tension: 300,
          friction: 3,
        }),
        Animated.spring(scale, {
          toValue: 1,
          useNativeDriver: true,
          tension: 300,
          friction: 3,
        }),
      ]).start();
    } catch (error) {
      console.error('Error toggling favorite:', error);
    }
  };

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={toggleFavorite}
      activeOpacity={0.7}
    >
      <Animated.Text
        style={[
          styles.icon,
          {
            fontSize: size,
            color: isFavorite ? theme.warning : theme.textTertiary,
            transform: [{ scale }],
          },
        ]}
      >
        {isFavorite ? '⭐' : '☆'}
      </Animated.Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: spacing.xs,
  },
  icon: {
    textAlign: 'center',
  },
});

