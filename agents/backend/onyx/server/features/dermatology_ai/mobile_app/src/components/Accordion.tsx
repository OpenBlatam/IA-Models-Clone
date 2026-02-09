import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface AccordionItem {
  id: string;
  title: string;
  content: React.ReactNode;
  icon?: keyof typeof Ionicons.glyphMap;
}

interface AccordionProps {
  items: AccordionItem[];
  allowMultiple?: boolean;
  defaultOpen?: string[];
}

const Accordion: React.FC<AccordionProps> = ({
  items,
  allowMultiple = false,
  defaultOpen = [],
}) => {
  const { colors } = useTheme();
  const [openItems, setOpenItems] = useState<string[]>(defaultOpen);

  const toggleItem = (id: string) => {
    setOpenItems((prev) => {
      if (prev.includes(id)) {
        return prev.filter((item) => item !== id);
      } else {
        return allowMultiple ? [...prev, id] : [id];
      }
    });
  };

  return (
    <View style={styles.container}>
      {items.map((item) => {
        const isOpen = openItems.includes(item.id);
        const rotation = useSharedValue(isOpen ? 1 : 0);

        React.useEffect(() => {
          rotation.value = withSpring(isOpen ? 1 : 0, {
            damping: 15,
            stiffness: 200,
          });
        }, [isOpen, rotation]);

        const animatedIconStyle = useAnimatedStyle(() => {
          return {
            transform: [{ rotate: `${rotation.value * 180}deg` }],
          };
        });

        return (
          <View
            key={item.id}
            style={[
              styles.item,
              {
                backgroundColor: colors.card,
                borderColor: colors.border,
              },
            ]}
          >
            <TouchableOpacity
              style={styles.header}
              onPress={() => toggleItem(item.id)}
              activeOpacity={0.7}
            >
              <View style={styles.headerContent}>
                {item.icon && (
                  <Ionicons
                    name={item.icon}
                    size={20}
                    color={colors.primary}
                    style={styles.icon}
                  />
                )}
                <Text style={[styles.title, { color: colors.text }]}>
                  {item.title}
                </Text>
              </View>
              <Animated.View style={animatedIconStyle}>
                <Ionicons
                  name="chevron-down"
                  size={20}
                  color={colors.textSecondary}
                />
              </Animated.View>
            </TouchableOpacity>
            {isOpen && (
              <View style={[styles.content, { borderTopColor: colors.border }]}>
                {item.content}
              </View>
            )}
          </View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    gap: 8,
  },
  item: {
    borderRadius: 12,
    borderWidth: 1,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  icon: {
    marginRight: 12,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  content: {
    padding: 16,
    borderTopWidth: 1,
  },
});

export default Accordion;

