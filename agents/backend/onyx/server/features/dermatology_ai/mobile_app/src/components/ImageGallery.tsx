import React, { useState } from 'react';
import {
  View,
  Image,
  Text,
  StyleSheet,
  TouchableOpacity,
  Modal,
  ScrollView,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface ImageGalleryProps {
  images: string[];
  columns?: number;
  onImagePress?: (index: number) => void;
}

const ImageGallery: React.FC<ImageGalleryProps> = ({
  images,
  columns = 3,
  onImagePress,
}) => {
  const { colors } = useTheme();
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const screenWidth = Dimensions.get('window').width;
  const imageSize = (screenWidth - 32 - (columns - 1) * 8) / columns;

  const handleImagePress = (index: number) => {
    setSelectedIndex(index);
    onImagePress?.(index);
  };

  const closeModal = () => {
    setSelectedIndex(null);
  };

  const navigateImage = (direction: 'prev' | 'next') => {
    if (selectedIndex === null) return;
    if (direction === 'prev' && selectedIndex > 0) {
      setSelectedIndex(selectedIndex - 1);
    } else if (direction === 'next' && selectedIndex < images.length - 1) {
      setSelectedIndex(selectedIndex + 1);
    }
  };

  return (
    <>
      <View style={styles.container}>
        {images.map((uri, index) => (
          <TouchableOpacity
            key={index}
            onPress={() => handleImagePress(index)}
            style={[
              styles.imageContainer,
              {
                width: imageSize,
                height: imageSize,
                marginRight: (index + 1) % columns !== 0 ? 8 : 0,
                marginBottom: 8,
              },
            ]}
            activeOpacity={0.8}
          >
            <Image
              source={{ uri }}
              style={styles.image}
              resizeMode="cover"
            />
            {index === images.length - 1 && images.length > columns && (
              <View style={styles.overlay}>
                <Text style={styles.overlayText}>+{images.length - columns}</Text>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>

      <Modal
        visible={selectedIndex !== null}
        transparent={true}
        animationType="fade"
        onRequestClose={closeModal}
      >
        <View style={styles.modalContainer}>
          <TouchableOpacity
            style={styles.closeButton}
            onPress={closeModal}
            activeOpacity={0.7}
          >
            <Ionicons name="close" size={32} color="#fff" />
          </TouchableOpacity>

          {selectedIndex !== null && (
            <>
              {selectedIndex > 0 && (
                <TouchableOpacity
                  style={[styles.navButton, styles.prevButton]}
                  onPress={() => navigateImage('prev')}
                  activeOpacity={0.7}
                >
                  <Ionicons name="chevron-back" size={32} color="#fff" />
                </TouchableOpacity>
              )}

              <ScrollView
                horizontal
                pagingEnabled
                showsHorizontalScrollIndicator={false}
                contentOffset={{ x: selectedIndex * screenWidth, y: 0 }}
              >
                {images.map((uri, index) => (
                  <Image
                    key={index}
                    source={{ uri }}
                    style={[styles.fullImage, { width: screenWidth }]}
                    resizeMode="contain"
                  />
                ))}
              </ScrollView>

              {selectedIndex < images.length - 1 && (
                <TouchableOpacity
                  style={[styles.navButton, styles.nextButton]}
                  onPress={() => navigateImage('next')}
                  activeOpacity={0.7}
                >
                  <Ionicons name="chevron-forward" size={32} color="#fff" />
                </TouchableOpacity>
              )}

              <View style={styles.counter}>
                <Text style={styles.counterText}>
                  {selectedIndex + 1} / {images.length}
                </Text>
              </View>
            </>
          )}
        </View>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginVertical: 8,
  },
  imageContainer: {
    borderRadius: 8,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlayText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
    padding: 8,
  },
  navButton: {
    position: 'absolute',
    top: '50%',
    transform: [{ translateY: -20 }],
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 25,
    padding: 8,
    zIndex: 10,
  },
  prevButton: {
    left: 20,
  },
  nextButton: {
    right: 20,
  },
  fullImage: {
    height: '100%',
  },
  counter: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  counterText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});

export default ImageGallery;

