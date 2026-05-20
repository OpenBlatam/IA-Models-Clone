import React, { useState, useEffect } from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { storage, STORAGE_KEYS } from '../utils/storage';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface FavoritesFilterProps {
  onToggle: (showFavorites: boolean) => void;
  initialValue?: boolean;
}

export const FavoritesFilter: React.FC<FavoritesFilterProps> = ({
  onToggle,
  initialValue = false,
}) => {
  const { theme } = useTheme();
  const [showFavorites, setShowFavorites] = useState(initialValue);
  const [favoritesCount, setFavoritesCount] = useState(0);

  useEffect(() => {
    loadFavoritesCount();
  }, []);

  const loadFavoritesCount = async () => {
    try {
      const favorites = await storage.get<string[]>(STORAGE_KEYS.FAVORITES) || [];
      setFavoritesCount(favorites.length);
    } catch (error) {
      console.error('Error loading favorites count:', error);
    }
  };

  const handleToggle = () => {
    hapticFeedback.selection();
    const newValue = !showFavorites;
    setShowFavorites(newValue);
    onToggle(newValue);
  };

  return (
    <TouchableOpacity
      style={[
        styles.container,
        {
          backgroundColor: showFavorites ? theme.primary : theme.surfaceVariant,
          borderColor: theme.border,
        },
      ]}
      onPress={handleToggle}
      activeOpacity={0.7}
    >
      <Text
        style={[
          styles.icon,
          { color: showFavorites ? theme.surface : theme.text },
        ]}
      >
        ⭐
      </Text>
      <Text
        style={[
          styles.text,
          { color: showFavorites ? theme.surface : theme.text },
        ]}
      >
        Favoritos
      </Text>
      {favoritesCount > 0 && (
        <View
          style={[
            styles.badge,
            {
              backgroundColor: showFavorites ? theme.surface : theme.primary,
            },
          ]}
        >
          <Text
            style={[
              styles.badgeText,
              {
                color: showFavorites ? theme.primary : theme.surface,
              },
            ]}
          >
            {favoritesCount}
          </Text>
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    gap: spacing.xs,
  },
  icon: {
    fontSize: 16,
  },
  text: {
    ...typography.bodySmall,
    fontWeight: '600',
  },
  badge: {
    minWidth: 20,
    height: 20,
    borderRadius: 10,
    paddingHorizontal: spacing.xs,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: spacing.xs,
  },
  badgeText: {
    ...typography.caption,
    fontWeight: '700',
    fontSize: 10,
  },
});

