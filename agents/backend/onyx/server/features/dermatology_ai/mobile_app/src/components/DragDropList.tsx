import React, { useState } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  Text,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface DragDropListProps<T> {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  onReorder: (newOrder: T[]) => void;
  keyExtractor: (item: T, index: number) => string;
}

function DragDropList<T>({
  data,
  renderItem,
  onReorder,
  keyExtractor,
}: DragDropListProps<T>) {
  const { colors } = useTheme();
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
  };

  const moveItem = (fromIndex: number, toIndex: number) => {
    const newData = [...data];
    const [removed] = newData.splice(fromIndex, 1);
    newData.splice(toIndex, 0, removed);
    onReorder(newData);
  };

  return (
    <FlatList
      data={data}
      keyExtractor={keyExtractor}
      renderItem={({ item, index }) => (
        <View
          style={[
            styles.item,
            {
              backgroundColor: colors.card,
              opacity: draggedIndex === index ? 0.5 : 1,
            },
          ]}
        >
          <TouchableOpacity
            style={styles.dragHandle}
            onLongPress={() => handleDragStart(index)}
            onPressOut={handleDragEnd}
          >
            <Ionicons name="reorder-three" size={24} color={colors.textSecondary} />
          </TouchableOpacity>
          <View style={styles.content}>{renderItem(item, index)}</View>
        </View>
      )}
      ItemSeparatorComponent={() => (
        <View style={[styles.separator, { backgroundColor: colors.border }]} />
      )}
    />
  );
}

const styles = StyleSheet.create({
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
  },
  dragHandle: {
    marginRight: 12,
    padding: 4,
  },
  content: {
    flex: 1,
  },
  separator: {
    height: 1,
    marginHorizontal: 12,
  },
});

export default DragDropList;

