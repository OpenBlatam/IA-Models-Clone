import React from 'react';
import { View, Text, Image, StyleSheet, ViewStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius } from '../theme/colors';

interface AvatarProps {
  source?: { uri: string };
  name?: string;
  size?: number;
  variant?: 'circle' | 'square' | 'rounded';
  style?: ViewStyle;
}

export const Avatar: React.FC<AvatarProps> = ({
  source,
  name,
  size = 40,
  variant = 'circle',
  style,
}) => {
  const { theme } = useTheme();

  const getInitials = (fullName?: string): string => {
    if (!fullName) return '?';
    const names = fullName.trim().split(' ');
    if (names.length === 1) {
      return names[0].charAt(0).toUpperCase();
    }
    return (names[0].charAt(0) + names[names.length - 1].charAt(0)).toUpperCase();
  };

  const getBorderRadius = () => {
    switch (variant) {
      case 'circle':
        return size / 2;
      case 'rounded':
        return size * 0.2;
      case 'square':
        return 0;
      default:
        return size / 2;
    }
  };

  const containerStyle = {
    width: size,
    height: size,
    borderRadius: getBorderRadius(),
    backgroundColor: theme.surfaceVariant,
  };

  if (source?.uri) {
    return (
      <Image
        source={source}
        style={[styles.image, containerStyle, style]}
        resizeMode="cover"
      />
    );
  }

  return (
    <View style={[styles.container, containerStyle, style]}>
      <Text
        style={[
          styles.initials,
          {
            color: theme.text,
            fontSize: size * 0.4,
          },
        ]}
      >
        {getInitials(name)}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  initials: {
    fontWeight: '600',
  },
});

