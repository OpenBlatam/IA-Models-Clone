import { useRef } from 'react';
import { ScrollView, FlatList } from 'react-native';

export const useScrollToTop = () => {
  const scrollViewRef = useRef<ScrollView | FlatList | null>(null);

  const scrollToTop = (animated: boolean = true) => {
    if (scrollViewRef.current) {
      if ('scrollTo' in scrollViewRef.current) {
        scrollViewRef.current.scrollTo({ y: 0, animated });
      } else if ('scrollToOffset' in scrollViewRef.current) {
        scrollViewRef.current.scrollToOffset({ offset: 0, animated });
      }
    }
  };

  return {
    scrollViewRef,
    scrollToTop,
  };
};

