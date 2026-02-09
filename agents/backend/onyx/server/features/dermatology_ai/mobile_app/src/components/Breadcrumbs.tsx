import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface BreadcrumbItem {
  label: string;
  onPress?: () => void;
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[];
  separator?: React.ReactNode;
}

const Breadcrumbs: React.FC<BreadcrumbsProps> = ({
  items,
  separator,
}) => {
  const { colors } = useTheme();

  const defaultSeparator = (
    <Ionicons
      name="chevron-forward"
      size={16}
      color={colors.textSecondary}
      style={styles.separator}
    />
  );

  return (
    <View style={styles.container}>
      {items.map((item, index) => {
        const isLast = index === items.length - 1;
        const Component = item.onPress && !isLast ? TouchableOpacity : View;

        return (
          <React.Fragment key={index}>
            <Component
              onPress={item.onPress}
              style={styles.item}
              disabled={isLast || !item.onPress}
            >
              <Text
                style={[
                  styles.label,
                  {
                    color: isLast ? colors.text : colors.primary,
                    fontWeight: isLast ? '600' : '400',
                  },
                ]}
              >
                {item.label}
              </Text>
            </Component>
            {!isLast && (separator || defaultSeparator)}
          </React.Fragment>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  item: {
    paddingVertical: 4,
  },
  label: {
    fontSize: 14,
  },
  separator: {
    marginHorizontal: 8,
  },
});

export default Breadcrumbs;

