import { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  Image,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { useMemes, useCreateMeme, useDeleteMeme, useMemeCategories } from '@/hooks/useApi';
import { ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { Meme } from '@/types';

export default function MemesScreen() {
  const [categoryFilter, setCategoryFilter] = useState<string | undefined>(undefined);
  const { data: memes, isLoading, refetch } = useMemes({ category: categoryFilter });
  const { data: categories } = useMemeCategories();
  const createMeme = useCreateMeme();
  const deleteMeme = useDeleteMeme();

  const handlePickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please grant camera roll permissions');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      const formData = new FormData();
      formData.append('file', {
        uri: result.assets[0].uri,
        type: 'image/jpeg',
        name: 'meme.jpg',
      } as any);
      formData.append('caption', '');
      formData.append('category', 'general');

      try {
        await createMeme.mutateAsync(formData);
        Alert.alert('Success', 'Meme uploaded successfully');
        refetch();
      } catch (error) {
        Alert.alert('Error', 'Failed to upload meme');
      }
    }
  };

  const handleDelete = (memeId: string) => {
    Alert.alert('Delete Meme', 'Are you sure you want to delete this meme?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            await deleteMeme.mutateAsync(memeId);
            Alert.alert('Success', 'Meme deleted');
            refetch();
          } catch (error) {
            Alert.alert('Error', 'Failed to delete meme');
          }
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.addButton} onPress={handlePickImage}>
          <Ionicons name="add" size={24} color="#fff" />
        </TouchableOpacity>
        {categories && categories.length > 0 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.categories}>
            <TouchableOpacity
              style={[styles.categoryButton, !categoryFilter && styles.categoryButtonActive]}
              onPress={() => setCategoryFilter(undefined)}
            >
              <Text style={[styles.categoryText, !categoryFilter && styles.categoryTextActive]}>
                All
              </Text>
            </TouchableOpacity>
            {categories.map((category) => (
              <TouchableOpacity
                key={category}
                style={[
                  styles.categoryButton,
                  categoryFilter === category && styles.categoryButtonActive,
                ]}
                onPress={() => setCategoryFilter(category)}
              >
                <Text
                  style={[
                    styles.categoryText,
                    categoryFilter === category && styles.categoryTextActive,
                  ]}
                >
                  {category}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        )}
      </View>

      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refetch} />}
      >
        {isLoading && !memes ? (
          <ActivityIndicator size="large" color="#0ea5e9" style={styles.loader} />
        ) : memes && memes.length > 0 ? (
          <View style={styles.grid}>
            {memes.map((meme) => (
              <MemeCard key={meme.meme_id} meme={meme} onDelete={() => handleDelete(meme.meme_id)} />
            ))}
          </View>
        ) : (
          <View style={styles.empty}>
            <Ionicons name="image-outline" size={64} color="#d1d5db" />
            <Text style={styles.emptyText}>No memes found</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function MemeCard({ meme, onDelete }: { meme: Meme; onDelete: () => void }) {
  return (
    <View style={styles.card}>
      <Image source={{ uri: meme.image_path }} style={styles.image} resizeMode="cover" />
      {meme.caption && <Text style={styles.caption} numberOfLines={2}>{meme.caption}</Text>}
      <View style={styles.tags}>
        {meme.tags.slice(0, 3).map((tag) => (
          <Text key={tag} style={styles.tag}>
            #{tag}
          </Text>
        ))}
      </View>
      <TouchableOpacity style={styles.deleteButton} onPress={onDelete}>
        <Ionicons name="trash" size={16} color="#ef4444" />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  addButton: {
    backgroundColor: '#0ea5e9',
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'flex-end',
    marginBottom: 12,
  },
  categories: {
    flexDirection: 'row',
  },
  categoryButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f3f4f6',
    marginRight: 8,
  },
  categoryButtonActive: {
    backgroundColor: '#0ea5e9',
  },
  categoryText: {
    fontSize: 14,
    color: '#6b7280',
  },
  categoryTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  scrollView: {
    flex: 1,
  },
  loader: {
    marginTop: 40,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 8,
  },
  card: {
    width: '48%',
    margin: '1%',
    backgroundColor: '#fff',
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  image: {
    width: '100%',
    height: 200,
  },
  caption: {
    padding: 8,
    fontSize: 12,
    color: '#4b5563',
  },
  tags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 8,
    gap: 4,
  },
  tag: {
    fontSize: 10,
    color: '#6b7280',
  },
  deleteButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 16,
    padding: 6,
  },
  empty: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    marginTop: 60,
  },
  emptyText: {
    fontSize: 16,
    color: '#9ca3af',
    marginTop: 16,
  },
});


