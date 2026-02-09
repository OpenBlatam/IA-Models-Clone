import * as ImageManipulator from 'expo-image-manipulator';

/**
 * Optimize image for analysis
 */
export const optimizeImage = async (
  uri: string,
  maxWidth: number = 1024,
  maxHeight: number = 1024,
  quality: number = 0.8
): Promise<string> => {
  try {
    const manipResult = await ImageManipulator.manipulateAsync(
      uri,
      [
        {
          resize: {
            width: maxWidth,
            height: maxHeight,
          },
        },
      ],
      {
        compress: quality,
        format: ImageManipulator.SaveFormat.JPEG,
      }
    );

    return manipResult.uri;
  } catch (error) {
    console.error('Error optimizing image:', error);
    return uri; // Return original if optimization fails
  }
};

/**
 * Compress image
 */
export const compressImage = async (
  uri: string,
  quality: number = 0.7
): Promise<string> => {
  try {
    const manipResult = await ImageManipulator.manipulateAsync(
      uri,
      [],
      {
        compress: quality,
        format: ImageManipulator.SaveFormat.JPEG,
      }
    );

    return manipResult.uri;
  } catch (error) {
    console.error('Error compressing image:', error);
    return uri;
  }
};

