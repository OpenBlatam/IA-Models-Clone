import React from 'react';
import { View, Text, Image, StyleSheet, ViewStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface AvatarProps {
  source?: { uri: string };
  name?: string;
  size?: number;
  variant?: 'circle' | 'square' | 'rounded';
  icon?: keyof typeof Ionicons.glyphMap;
  showBadge?: boolean;
  badgeColor?: string;
  style?: ViewStyle;
}

const Avatar: React.FC<AvatarProps> = ({
  source,
  name,
  size = 48,
  variant = 'circle',
  icon,
  showBadge = false,
  badgeColor,
  style,
}) => {
  const { colors } = useTheme();

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map((n) => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const borderRadius = variant === 'circle' ? size / 2 : variant === 'rounded' ? size / 4 : 0;

  const renderContent = () => {
    if (source) {
      return (
        <Image
          source={source}
          style={[
            styles.image,
            {
              width: size,
              height: size,
              borderRadius,
            },
          ]}
        />
      );
    }

    if (icon) {
      return (
        <View
          style={[
            styles.iconContainer,
            {
              width: size,
              height: size,
              borderRadius,
              backgroundColor: colors.primary,
            },
          ]}
        >
          <Ionicons name={icon} size={size * 0.5} color="#fff" />
        </View>
      );
    }

    if (name) {
      return (
        <LinearGradient
          colors={[colors.primary, colors.secondary]}
          style={[
            styles.initialsContainer,
            {
              width: size,
              height: size,
              borderRadius,
            },
          ]}
        >
          <Text
            style={[
              styles.initials,
              {
                fontSize: size * 0.4,
                color: '#fff',
              },
            ]}
          >
            {getInitials(name)}
          </Text>
        </LinearGradient>
      );
    }

    return (
      <View
        style={[
          styles.placeholder,
          {
            width: size,
            height: size,
            borderRadius,
            backgroundColor: colors.border,
          },
        ]}
      >
        <Ionicons name="person" size={size * 0.5} color={colors.textSecondary} />
      </View>
    );
  };

  return (
    <View style={[styles.container, style]}>
      {renderContent()}
      {showBadge && (
        <View
          style={[
            styles.badge,
            {
              width: size * 0.3,
              height: size * 0.3,
              borderRadius: size * 0.15,
              backgroundColor: badgeColor || colors.success,
              borderColor: colors.background,
              borderWidth: 2,
            },
          ]}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  image: {
    resizeMode: 'cover',
  },
  iconContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  initialsContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  initials: {
    fontWeight: '600',
  },
  placeholder: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  badge: {
    position: 'absolute',
    bottom: 0,
    right: 0,
  },
});

export default Avatar;

