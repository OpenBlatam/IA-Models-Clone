import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  action?: {
    label: string;
    onPress: () => void;
  };
  icon?: keyof typeof Ionicons.glyphMap;
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
  action,
  icon,
}) => {
  const { colors } = useTheme();

  return (
    <View style={styles.container}>
      <View style={styles.left}>
        {icon && (
          <Ionicons
            name={icon}
            size={20}
            color={colors.primary}
            style={styles.icon}
          />
        )}
        <View>
          <Text style={[styles.title, { color: colors.text }]}>{title}</Text>
          {subtitle && (
            <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
              {subtitle}
            </Text>
          )}
        </View>
      </View>
      {action && (
        <TouchableOpacity onPress={action.onPress} activeOpacity={0.7}>
          <Text style={[styles.action, { color: colors.primary }]}>
            {action.label}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  left: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  icon: {
    marginRight: 12,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
  },
  subtitle: {
    fontSize: 14,
    marginTop: 2,
  },
  action: {
    fontSize: 14,
    fontWeight: '600',
  },
});

export default SectionHeader;

