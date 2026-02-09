import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { Collapsible } from './Collapsible';
import { spacing, borderRadius, typography } from '../theme/colors';

interface AccordionItem {
  id: string;
  title: string;
  content: React.ReactNode;
  icon?: React.ReactNode;
}

interface AccordionProps {
  items: AccordionItem[];
  allowMultiple?: boolean;
  defaultExpanded?: string[];
}

export const Accordion: React.FC<AccordionProps> = ({
  items,
  allowMultiple = false,
  defaultExpanded = [],
}) => {
  const { theme } = useTheme();
  const [expandedItems, setExpandedItems] = useState<string[]>(defaultExpanded);

  const handleToggle = (itemId: string) => {
    setExpandedItems((prev) => {
      if (allowMultiple) {
        return prev.includes(itemId)
          ? prev.filter((id) => id !== itemId)
          : [...prev, itemId];
      }
      return prev.includes(itemId) ? [] : [itemId];
    });
  };

  return (
    <View style={styles.container}>
      {items.map((item) => (
        <Collapsible
          key={item.id}
          title={item.title}
          defaultExpanded={expandedItems.includes(item.id)}
          icon={item.icon}
          onToggle={() => handleToggle(item.id)}
        >
          {item.content}
        </Collapsible>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    gap: spacing.sm,
  },
});

