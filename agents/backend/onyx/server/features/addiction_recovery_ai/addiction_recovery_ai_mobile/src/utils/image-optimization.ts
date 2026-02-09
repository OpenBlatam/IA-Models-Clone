import { ImageSourcePropType } from 'react-native';

export interface OptimizedImageSource {
  uri: string;
  width?: number;
  height?: number;
  cache?: 'default' | 'reload' | 'force-cache' | 'only-if-cached';
}

export function optimizeImageSource(
  source: ImageSourcePropType,
  width?: number,
  height?: number
): OptimizedImageSource {
  if (typeof source === 'number') {
    return source as unknown as OptimizedImageSource;
  }

  if (typeof source === 'object' && 'uri' in source) {
    return {
      uri: source.uri,
      width,
      height,
      cache: 'force-cache',
    };
  }

  return source as OptimizedImageSource;
}

export function getImageDimensions(
  uri: string
): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const Image = require('react-native').Image;
    Image.getSize(
      uri,
      (width: number, height: number) => {
        resolve({ width, height });
      },
      (error: Error) => {
        reject(error);
      }
    );
  });
}

