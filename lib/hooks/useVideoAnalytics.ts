import { useState, useEffect, useCallback } from 'react';
import { VideoAnalytics } from '@/lib/types/video';

interface UseVideoAnalyticsOptions {
  videoId: string;
  userId?: string;
  trackingInterval?: number;
}

export function useVideoAnalytics({
  videoId,
  userId,
  trackingInterval = 10000, // 10 seconds
}: UseVideoAnalyticsOptions) {
  const [analytics, setAnalytics] = useState<VideoAnalytics | null>(null);
  const [sessionStartTime, setSessionStartTime] = useState<number>(Date.now());
  const [totalWatchTime, setTotalWatchTime] = useState<number>(0);
  const [lastTrackingTime, setLastTrackingTime] = useState<number>(Date.now());

  const trackWatchTime = useCallback((currentTime: number, isPlaying: boolean) => {
    if (!isPlaying) return;

    const now = Date.now();
    const timeDiff = now - lastTrackingTime;
    
    if (timeDiff >= trackingInterval) {
      setTotalWatchTime(prev => prev + timeDiff);
      setLastTrackingTime(now);
      
      if (userId) {
        fetch('/api/analytics/video-watch', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            videoId,
            userId,
            watchTime: timeDiff,
            currentTime,
            timestamp: now,
          }),
        }).catch(error => {
          console.error('Failed to track video analytics:', error);
        });
      }
    }
  }, [videoId, userId, trackingInterval, lastTrackingTime]);

  const trackVideoEvent = useCallback((event: string, data?: any) => {
    if (!userId) return;

    fetch('/api/analytics/video-event', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        videoId,
        userId,
        event,
        data,
        timestamp: Date.now(),
      }),
    }).catch(error => {
      console.error('Failed to track video event:', error);
    });
  }, [videoId, userId]);

  const fetchAnalytics = useCallback(async () => {
    try {
      const response = await fetch(`/api/analytics/video/${videoId}`);
      if (response.ok) {
        const data = await response.json();
        setAnalytics(data);
      }
    } catch (error) {
      console.error('Failed to fetch video analytics:', error);
    }
  }, [videoId]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  useEffect(() => {
    setSessionStartTime(Date.now());
    setLastTrackingTime(Date.now());
    setTotalWatchTime(0);
  }, [videoId]);

  return {
    analytics,
    trackWatchTime,
    trackVideoEvent,
    totalWatchTime,
    sessionStartTime,
    refreshAnalytics: fetchAnalytics,
  };
}
