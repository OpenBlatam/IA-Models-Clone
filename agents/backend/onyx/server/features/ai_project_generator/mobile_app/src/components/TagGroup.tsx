import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Tag } from './Tag';
import { spacing } from '../theme/colors';

interface TagGroupProps {
  tags: Array<{
    id: string;
    label: string;
    variant?: 'default' | 'primary' | 'success' | 'warning' | 'error';
    onPress?: () => void;
    onClose?: () => void;
  }>;
  onTagPress?: (id: string) => void;
  onTagClose?: (id: string) => void;
  wrap?: boolean;
}

export const TagGroup: React.FC<TagGroupProps> = ({
  tags,
  onTagPress,
  onTagClose,
  wrap = true,
}) => {
  return (
    <View style={[styles.container, !wrap && styles.row]}>
      {tags.map((tag, index) => (
        <View
          key={tag.id}
          style={[
            styles.tagWrapper,
            index < tags.length - 1 && styles.tagSpacing,
          ]}
        >
          <Tag
            label={tag.label}
            variant={tag.variant}
            onPress={tag.onPress || (onTagPress ? () => onTagPress(tag.id) : undefined)}
            onClose={tag.onClose || (onTagClose ? () => onTagClose(tag.id) : undefined)}
          />
        </View>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  row: {
    flexWrap: 'nowrap',
  },
  tagWrapper: {
    marginBottom: spacing.xs,
  },
  tagSpacing: {
    marginRight: spacing.sm,
  },
});

