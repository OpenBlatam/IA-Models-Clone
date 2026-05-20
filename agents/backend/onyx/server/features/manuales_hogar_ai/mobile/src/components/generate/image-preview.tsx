/**
 * Image Preview
 * =============
 * Component for previewing selected images
 */

import { View, Image, StyleSheet, TouchableOpacity, ScrollView, Text } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';

interface ImagePreviewProps {
  images: string[];
  onRemove: (index: number) => void;
  maxImages?: number;
}

export function ImagePreview({ images, onRemove, maxImages = 5 }: ImagePreviewProps) {
  const { state } = useApp();
  const colors = state.colors;

  if (images.length === 0) return null;

  return (
    <View style={styles.container}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.scrollView}>
        {images.map((uri, index) => (
          <View key={index} style={[styles.imageContainer, { backgroundColor: colors.card }]}>
            <Image source={{ uri }} style={styles.image} resizeMode="cover" />
            <TouchableOpacity
              style={[styles.removeButton, { backgroundColor: colors.error }]}
              onPress={() => onRemove(index)}
            >
              <Ionicons name="close" size={16} color="#FFFFFF" />
            </TouchableOpacity>
            {index === 0 && (
              <View style={[styles.badge, { backgroundColor: colors.tint }]}>
                <Text style={styles.badgeText}>Main</Text>
              </View>
            )}
          </View>
        ))}
        {images.length < maxImages && (
          <View style={[styles.addContainer, { borderColor: colors.border }]}>
            <Ionicons name="add" size={32} color={colors.textSecondary} />
            <Text style={[styles.addText, { color: colors.textSecondary }]}>
              Add more
            </Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: 16,
  },
  scrollView: {
    paddingHorizontal: 20,
  },
  imageContainer: {
    width: 120,
    height: 120,
    borderRadius: 12,
    marginRight: 12,
    position: 'relative',
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  removeButton: {
    position: 'absolute',
    top: 4,
    right: 4,
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  badge: {
    position: 'absolute',
    bottom: 4,
    left: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  badgeText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '600',
  },
  addContainer: {
    width: 120,
    height: 120,
    borderRadius: 12,
    borderWidth: 2,
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  addText: {
    fontSize: 12,
    marginTop: 4,
  },
});

