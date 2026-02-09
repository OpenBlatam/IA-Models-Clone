import { memo, useState, useCallback } from 'react';
import { useSpeechRecognition } from '@/lib/hooks';
import Button from './Button';
import { Mic, MicOff, Square } from 'lucide-react';
import { cn } from '@/lib/utils';

interface VoiceRecorderProps {
  onTranscript?: (transcript: string) => void;
  onError?: (error: string) => void;
  className?: string;
  continuous?: boolean;
}

const VoiceRecorder = memo(({
  onTranscript,
  onError,
  className = '',
  continuous = false,
}: VoiceRecorderProps): JSX.Element => {
  const { isListening, transcript, error, start, stop, supported } = useSpeechRecognition({
    continuous,
    onResult: onTranscript,
    onError: (err) => {
      if (onError) {
        onError(err.error);
      }
    },
  });

  const handleToggle = useCallback(() => {
    if (isListening) {
      stop();
    } else {
      start();
    }
  }, [isListening, start, stop]);

  if (!supported) {
    return (
      <div className={cn('text-sm text-gray-500', className)}>
        Speech recognition is not supported in your browser
      </div>
    );
  }

  return (
    <div className={cn('space-y-2', className)}>
      <Button
        onClick={handleToggle}
        variant={isListening ? 'danger' : 'primary'}
        className="flex items-center gap-2"
        aria-label={isListening ? 'Stop recording' : 'Start recording'}
      >
        {isListening ? (
          <>
            <Square className="w-4 h-4" />
            Stop
          </>
        ) : (
          <>
            <Mic className="w-4 h-4" />
            Start Recording
          </>
        )}
      </Button>

      {isListening && (
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <MicOff className="w-4 h-4 animate-pulse" />
          <span>Listening...</span>
        </div>
      )}

      {transcript && (
        <div className="p-3 bg-gray-50 rounded text-sm">
          <p className="font-semibold mb-1">Transcript:</p>
          <p>{transcript}</p>
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-600">
          Error: {error}
        </div>
      )}
    </div>
  );
});

VoiceRecorder.displayName = 'VoiceRecorder';

export default VoiceRecorder;



