import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Image } from 'expo-image';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { formatInitials } from '../../utils/format-helpers';

interface AvatarProps {
  uri?: string;
  name?: string;
  size?: number;
  backgroundColor?: string;
  textColor?: string;
}

/**
 * Avatar component with fallback to initials
 */
export function Avatar({
  uri,
  name,
  size = 40,
  backgroundColor = COLORS.primary,
  textColor = COLORS.text,
}: AvatarProps) {
  const initials = name ? formatInitials(name, 2) : '?';
  const fontSize = size * 0.4;

  if (uri) {
    return (
      <Image
        source={{ uri }}
        style={[
          styles.container,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
          },
        ]}
        contentFit="cover"
      />
    );
  }

  return (
    <View
      style={[
        styles.container,
        styles.placeholder,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor,
        },
      ]}
    >
      <Text
        style={[
          styles.initials,
          {
            fontSize,
            color: textColor,
          },
        ]}
      >
        {initials}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholder: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  initials: {
    ...TYPOGRAPHY.body,
    fontWeight: '600',
  },
});

