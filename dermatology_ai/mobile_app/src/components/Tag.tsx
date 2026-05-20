import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface TagProps {
  label: string;
  onRemove?: () => void;
  color?: string;
  variant?: 'default' | 'outline' | 'filled';
  size?: 'small' | 'medium' | 'large';
  icon?: keyof typeof Ionicons.glyphMap;
}

const Tag: React.FC<TagProps> = ({
  label,
  onRemove,
  color,
  variant = 'default',
  size = 'medium',
  icon,
}) => {
  const { colors } = useTheme();
  const tagColor = color || colors.primary;

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { paddingVertical: 4, paddingHorizontal: 8, fontSize: 12 };
      case 'large':
        return { paddingVertical: 10, paddingHorizontal: 16, fontSize: 16 };
      default:
        return { paddingVertical: 6, paddingHorizontal: 12, fontSize: 14 };
    }
  };

  const sizeStyles = getSizeStyles();

  const getVariantStyles = () => {
    switch (variant) {
      case 'outline':
        return {
          backgroundColor: 'transparent',
          borderWidth: 1,
          borderColor: tagColor,
        };
      case 'filled':
        return {
          backgroundColor: tagColor,
        };
      default:
        return {
          backgroundColor: `${tagColor}20`,
        };
    }
  };

  const getTextColor = () => {
    if (variant === 'filled') return '#fff';
    return tagColor;
  };

  return (
    <View
      style={[
        styles.container,
        sizeStyles,
        getVariantStyles(),
      ]}
    >
      {icon && (
        <Ionicons
          name={icon}
          size={sizeStyles.fontSize}
          color={getTextColor()}
          style={styles.icon}
        />
      )}
      <Text
        style={[
          styles.label,
          {
            color: getTextColor(),
            fontSize: sizeStyles.fontSize,
          },
        ]}
      >
        {label}
      </Text>
      {onRemove && (
        <TouchableOpacity
          onPress={onRemove}
          style={styles.removeButton}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Ionicons
            name="close"
            size={sizeStyles.fontSize}
            color={getTextColor()}
          />
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 16,
    alignSelf: 'flex-start',
    marginRight: 8,
    marginBottom: 8,
  },
  label: {
    fontWeight: '500',
  },
  icon: {
    marginRight: 4,
  },
  removeButton: {
    marginLeft: 4,
    padding: 2,
  },
});

export default Tag;

