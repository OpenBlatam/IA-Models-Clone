/**
 * Analysis tab component.
 * Displays track analysis results with proper type guards and error handling.
 */

import { type Track, type MusicAnalysisResponse } from '@/lib/api/types';
import { TrackAnalysis } from '@/components/music/TrackAnalysis';
import { Sparkles, AlertCircle } from 'lucide-react';
import { LoadingState } from '@/components/ui';
import { Suspense } from 'react';
import dynamic from 'next/dynamic';

// Lazy load TrackAnalysis for better code splitting
const LazyTrackAnalysis = dynamic(
  () =>
    import('@/components/music/TrackAnalysis').then((mod) => ({
      default: mod.TrackAnalysis,
    })),
  {
    ssr: false,
    loading: () => <LoadingState message="Cargando análisis..." />,
  }
);

interface AnalysisTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Type guard to check if data is a valid MusicAnalysisResponse.
 *
 * @param data - Data to check
 * @returns True if data is valid MusicAnalysisResponse
 */
function isValidAnalysisData(
  data: unknown
): data is MusicAnalysisResponse {
  return (
    typeof data === 'object' &&
    data !== null &&
    'success' in data &&
    'track_basic_info' in data &&
    typeof (data as Record<string, unknown>).success === 'boolean'
  );
}

/**
 * Empty state component when no track is selected.
 */
function EmptyAnalysisState() {
  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-12 border border-white/20 text-center">
      <Sparkles className="w-16 h-16 text-purple-300 mx-auto mb-4" />
      <h3 className="text-xl font-semibold text-white mb-2">
        Selecciona una canción
      </h3>
      <p className="text-gray-300 text-lg">
        Busca y selecciona una canción para ver su análisis completo
      </p>
    </div>
  );
}

/**
 * Error state component for invalid analysis data.
 */
function InvalidAnalysisState() {
  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-12 border border-white/20 text-center">
      <AlertCircle className="w-16 h-16 text-red-300 mx-auto mb-4" />
      <h3 className="text-xl font-semibold text-white mb-2">
        Error en los datos
      </h3>
      <p className="text-gray-300 text-lg">
        Los datos de análisis no son válidos. Por favor, intenta de nuevo.
      </p>
    </div>
  );
}

/**
 * Analysis tab component.
 * Displays track analysis with proper error handling and type safety.
 *
 * @param props - Component props
 * @returns Analysis tab component
 */
export default function AnalysisTab({
  selectedTrack,
  analysisData,
}: AnalysisTabProps) {
  // Early return for missing track
  if (!selectedTrack) {
    return <EmptyAnalysisState />;
  }

  // Early return for missing analysis data
  if (!analysisData) {
    return <EmptyAnalysisState />;
  }

  // Type guard validation
  if (!isValidAnalysisData(analysisData)) {
    return <InvalidAnalysisState />;
  }

  return (
    <div className="space-y-6">
      <Suspense fallback={<LoadingState message="Cargando análisis..." />}>
        <LazyTrackAnalysis
          analysis={analysisData}
          track={selectedTrack}
        />
      </Suspense>
    </div>
  );
}
