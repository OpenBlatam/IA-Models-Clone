import { View, StyleSheet } from 'react-native';
import { SkeletonLoader } from './SkeletonLoader';

interface ContentLoaderProps {
  lines?: number;
  showAvatar?: boolean;
  avatarSize?: number;
}

export function ContentLoader({ lines = 3, showAvatar = false, avatarSize = 40 }: ContentLoaderProps) {
  return (
    <View style={styles.container}>
      {showAvatar && (
        <SkeletonLoader width={avatarSize} height={avatarSize} borderRadius={avatarSize / 2} />
      )}
      <View style={styles.content}>
        {Array.from({ length: lines }).map((_, index) => (
          <SkeletonLoader
            key={index}
            width={index === lines - 1 ? '80%' : '100%'}
            height={16}
            style={index > 0 ? styles.line : undefined}
          />
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  content: {
    flex: 1,
    gap: 8,
  },
  line: {
    marginTop: 8,
  },
});


