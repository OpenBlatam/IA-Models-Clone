import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface ExpandableTextProps {
  text: string;
  maxLength?: number;
  expandText?: string;
  collapseText?: string;
}

const ExpandableText: React.FC<ExpandableTextProps> = ({
  text,
  maxLength = 100,
  expandText = 'Ver más',
  collapseText = 'Ver menos',
}) => {
  const { colors } = useTheme();
  const [isExpanded, setIsExpanded] = useState(false);
  const shouldTruncate = text.length > maxLength;

  if (!shouldTruncate) {
    return <Text style={[styles.text, { color: colors.text }]}>{text}</Text>;
  }

  return (
    <View>
      <Text style={[styles.text, { color: colors.text }]}>
        {isExpanded ? text : `${text.substring(0, maxLength)}...`}
      </Text>
      <TouchableOpacity onPress={() => setIsExpanded(!isExpanded)}>
        <Text style={[styles.toggle, { color: colors.primary }]}>
          {isExpanded ? collapseText : expandText}
        </Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  text: {
    fontSize: 14,
    lineHeight: 20,
  },
  toggle: {
    fontSize: 14,
    fontWeight: '600',
    marginTop: 4,
  },
});

export default ExpandableText;

