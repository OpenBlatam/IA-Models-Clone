import { useState, useCallback } from 'react';

export const useImageOptimization = () => {
  const [imageError, setImageError] = useState(false);

  const handleImageError = useCallback(() => {
    setImageError(true);
  }, []);

  const handleImageLoad = useCallback(() => {
    setImageError(false);
  }, []);

  return {
    imageError,
    handleImageError,
    handleImageLoad,
  };
};

