/**
 * Featured Categories
 * ===================
 * Featured categories carousel
 */

import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import { useTranslation } from '@/hooks/use-translation';
import { CATEGORIES } from '@/constants/categories';
import Animated, { FadeInRight } from 'react-native-reanimated';

const { width } = Dimensions.get('window');
const CARD_WIDTH = width * 0.7;

export function FeaturedCategories() {
  const router = useRouter();
  const { state } = useApp();
  const { t } = useTranslation();
  const colors = state.colors;

  const featuredCategories = ['plomeria', 'electricidad', 'carpinteria', 'pintura'];

  return (
    <View style={styles.container}>
      <Text style={[styles.title, { color: colors.text }]}>{t('home.categories')}</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
        snapToInterval={CARD_WIDTH + 16}
        decelerationRate="fast"
        pagingEnabled
      >
        {featuredCategories.map((categoryId, index) => {
          const category = CATEGORIES[categoryId];
          if (!category) return null;

          return (
            <Animated.View
              key={categoryId}
              entering={FadeInRight.delay(index * 100)}
              style={styles.animationContainer}
            >
              <TouchableOpacity
                style={[
                  styles.card,
                  {
                    backgroundColor: category.color,
                    width: CARD_WIDTH,
                  },
                ]}
                onPress={() => router.push(`/(tabs)/generate?category=${categoryId}` as any)}
                activeOpacity={0.8}
              >
                <View style={styles.iconContainer}>
                  <Ionicons name={category.icon as any} size={48} color="#FFFFFF" />
                </View>
                <Text style={styles.categoryName}>{category.displayName}</Text>
                <Text style={styles.categoryDescription}>{category.description}</Text>
              </TouchableOpacity>
            </Animated.View>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: 24,
    marginBottom: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 12,
    paddingHorizontal: 20,
  },
  scrollContent: {
    paddingHorizontal: 20,
    gap: 16,
  },
  animationContainer: {
    marginRight: 16,
  },
  card: {
    padding: 24,
    borderRadius: 16,
    minHeight: 180,
    justifyContent: 'center',
    alignItems: 'center',
  },
  iconContainer: {
    marginBottom: 12,
  },
  categoryName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 8,
    textAlign: 'center',
  },
  categoryDescription: {
    fontSize: 14,
    color: '#FFFFFF',
    opacity: 0.9,
    textAlign: 'center',
  },
});



