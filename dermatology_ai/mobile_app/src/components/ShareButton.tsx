import React from 'react';
import { TouchableOpacity, StyleSheet, Share, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface ShareButtonProps {
  content: {
    message: string;
    title?: string;
    url?: string;
  };
  iconSize?: number;
  iconColor?: string;
}

const ShareButton: React.FC<ShareButtonProps> = ({
  content,
  iconSize = 24,
  iconColor = '#6366f1',
}) => {
  const handleShare = async () => {
    try {
      const result = await Share.share({
        message: content.message,
        title: content.title,
        url: content.url,
      });

      if (result.action === Share.sharedAction) {
        if (result.activityType) {
          // Shared with activity type of result.activityType
        } else {
          // Shared
        }
      } else if (result.action === Share.dismissedAction) {
        // Dismissed
      }
    } catch (error: any) {
      Alert.alert('Error', 'No se pudo compartir el contenido');
    }
  };

  return (
    <TouchableOpacity
      style={styles.button}
      onPress={handleShare}
      activeOpacity={0.7}
    >
      <Ionicons name="share-social" size={iconSize} color={iconColor} />
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    padding: 8,
  },
});

export default ShareButton;

