import { useState, useCallback, useEffect } from 'react';
import { PlaylistItem } from '@/lib/types/video';

interface UseVideoPlaylistOptions {
  initialPlaylist: PlaylistItem[];
  autoAdvance?: boolean;
  shuffle?: boolean;
  repeat?: 'none' | 'one' | 'all';
  onVideoChange?: (video: PlaylistItem, index: number) => void;
}

export function useVideoPlaylist({
  initialPlaylist,
  autoAdvance = true,
  shuffle = false,
  repeat = 'none',
  onVideoChange,
}: UseVideoPlaylistOptions) {
  const [playlist, setPlaylist] = useState<PlaylistItem[]>(initialPlaylist);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playHistory, setPlayHistory] = useState<number[]>([]);
  const [shuffleOrder, setShuffleOrder] = useState<number[]>([]);

  const currentVideo = playlist[currentIndex] || null;

  const generateShuffleOrder = useCallback(() => {
    const indices = Array.from({ length: playlist.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    setShuffleOrder(indices);
  }, [playlist.length]);

  useEffect(() => {
    if (shuffle) {
      generateShuffleOrder();
    }
  }, [shuffle, generateShuffleOrder]);

  const getNextIndex = useCallback(() => {
    if (shuffle) {
      const currentShuffleIndex = shuffleOrder.indexOf(currentIndex);
      const nextShuffleIndex = currentShuffleIndex + 1;
      
      if (nextShuffleIndex < shuffleOrder.length) {
        return shuffleOrder[nextShuffleIndex];
      } else if (repeat === 'all') {
        return shuffleOrder[0];
      }
      return -1;
    } else {
      const nextIndex = currentIndex + 1;
      if (nextIndex < playlist.length) {
        return nextIndex;
      } else if (repeat === 'all') {
        return 0;
      }
      return -1;
    }
  }, [currentIndex, shuffle, shuffleOrder, repeat, playlist.length]);

  const getPreviousIndex = useCallback(() => {
    if (shuffle) {
      const currentShuffleIndex = shuffleOrder.indexOf(currentIndex);
      const prevShuffleIndex = currentShuffleIndex - 1;
      
      if (prevShuffleIndex >= 0) {
        return shuffleOrder[prevShuffleIndex];
      } else if (repeat === 'all') {
        return shuffleOrder[shuffleOrder.length - 1];
      }
      return -1;
    } else {
      const prevIndex = currentIndex - 1;
      if (prevIndex >= 0) {
        return prevIndex;
      } else if (repeat === 'all') {
        return playlist.length - 1;
      }
      return -1;
    }
  }, [currentIndex, shuffle, shuffleOrder, repeat, playlist.length]);

  const playVideo = useCallback((index: number) => {
    if (index >= 0 && index < playlist.length) {
      setCurrentIndex(index);
      setPlayHistory(prev => [...prev, index]);
      onVideoChange?.(playlist[index], index);
    }
  }, [playlist, onVideoChange]);

  const playNext = useCallback(() => {
    const nextIndex = getNextIndex();
    if (nextIndex !== -1) {
      playVideo(nextIndex);
      return true;
    }
    return false;
  }, [getNextIndex, playVideo]);

  const playPrevious = useCallback(() => {
    const prevIndex = getPreviousIndex();
    if (prevIndex !== -1) {
      playVideo(prevIndex);
      return true;
    }
    return false;
  }, [getPreviousIndex, playVideo]);

  const handleVideoEnded = useCallback(() => {
    if (repeat === 'one') {
      onVideoChange?.(currentVideo!, currentIndex);
    } else if (autoAdvance) {
      playNext();
    }
  }, [repeat, autoAdvance, playNext, currentVideo, currentIndex, onVideoChange]);

  const addToPlaylist = useCallback((video: PlaylistItem, position?: number) => {
    setPlaylist(prev => {
      const newPlaylist = [...prev];
      if (position !== undefined && position >= 0 && position <= prev.length) {
        newPlaylist.splice(position, 0, video);
      } else {
        newPlaylist.push(video);
      }
      return newPlaylist;
    });
  }, []);

  const removeFromPlaylist = useCallback((index: number) => {
    setPlaylist(prev => {
      const newPlaylist = prev.filter((_, i) => i !== index);
      
      if (index < currentIndex) {
        setCurrentIndex(prev => prev - 1);
      } else if (index === currentIndex && currentIndex >= newPlaylist.length) {
        setCurrentIndex(Math.max(0, newPlaylist.length - 1));
      }
      
      return newPlaylist;
    });
  }, [currentIndex]);

  const moveInPlaylist = useCallback((fromIndex: number, toIndex: number) => {
    setPlaylist(prev => {
      const newPlaylist = [...prev];
      const [movedItem] = newPlaylist.splice(fromIndex, 1);
      newPlaylist.splice(toIndex, 0, movedItem);
      
      if (fromIndex === currentIndex) {
        setCurrentIndex(toIndex);
      } else if (fromIndex < currentIndex && toIndex >= currentIndex) {
        setCurrentIndex(prev => prev - 1);
      } else if (fromIndex > currentIndex && toIndex <= currentIndex) {
        setCurrentIndex(prev => prev + 1);
      }
      
      return newPlaylist;
    });
  }, [currentIndex]);

  const clearPlaylist = useCallback(() => {
    setPlaylist([]);
    setCurrentIndex(0);
    setPlayHistory([]);
  }, []);

  const updateVideoProgress = useCallback((videoId: string, progress: number) => {
    setPlaylist(prev => 
      prev.map(video => 
        video.id === videoId 
          ? { ...video, progress }
          : video
      )
    );
  }, []);

  const markVideoCompleted = useCallback((videoId: string) => {
    setPlaylist(prev => 
      prev.map(video => 
        video.id === videoId 
          ? { ...video, isCompleted: true, progress: 100 }
          : video
      )
    );
  }, []);

  return {
    playlist,
    currentVideo,
    currentIndex,
    playHistory,
    hasNext: getNextIndex() !== -1,
    hasPrevious: getPreviousIndex() !== -1,
    actions: {
      playVideo,
      playNext,
      playPrevious,
      handleVideoEnded,
      addToPlaylist,
      removeFromPlaylist,
      moveInPlaylist,
      clearPlaylist,
      updateVideoProgress,
      markVideoCompleted,
      generateShuffleOrder,
    },
  };
}
