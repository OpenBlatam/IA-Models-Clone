import React from 'react';
import { useRouter } from 'expo-router';
import { SearchScreen } from '../../components/music/search-screen';
import type { Track } from '../../types/api';

export default function SearchTab() {
  const router = useRouter();

  const handleTrackSelect = (track: Track) => {
    router.push({
      pathname: '/analysis',
      params: {
        trackId: track.id,
        trackName: track.name,
        artists: JSON.stringify(track.artists),
      },
    });
  };

  return <SearchScreen onTrackSelect={handleTrackSelect} />;
}

