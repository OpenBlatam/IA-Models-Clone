import React from 'react';
import { useLocalSearchParams } from 'expo-router';
import { AnalysisScreen } from '../components/music/analysis-screen';
import type { Track } from '../types/api';

export default function AnalysisPage() {
  const params = useLocalSearchParams<{
    trackId: string;
    trackName: string;
    artists: string;
  }>();

  const track: Track = {
    id: params.trackId || '',
    name: params.trackName || '',
    artists: params.artists ? JSON.parse(params.artists) : [],
    album: '',
    duration_ms: 0,
    preview_url: null,
    external_urls: {
      spotify: '',
    },
    popularity: 0,
  };

  return <AnalysisScreen track={track} />;
}

