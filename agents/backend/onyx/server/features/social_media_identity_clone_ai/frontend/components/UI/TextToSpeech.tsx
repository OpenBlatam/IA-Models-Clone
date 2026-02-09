import { memo, useState } from 'react';
import { useSpeechSynthesis } from '@/lib/hooks';
import Button from './Button';
import Select from './Select';
import { Volume2, VolumeX, Pause, Play } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TextToSpeechProps {
  text: string;
  className?: string;
  autoPlay?: boolean;
}

const TextToSpeech = memo(({
  text,
  className = '',
  autoPlay = false,
}: TextToSpeechProps): JSX.Element => {
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const { isSpeaking, speak, cancel, pause, resume, voices, supported } = useSpeechSynthesis({
    voice: selectedVoice || undefined,
  });

  const handleSpeak = (): void => {
    if (isSpeaking) {
      cancel();
    } else {
      speak(text);
    }
  };

  if (!supported) {
    return (
      <div className={cn('text-sm text-gray-500', className)}>
        Text-to-speech is not supported in your browser
      </div>
    );
  }

  const voiceOptions = voices.map((voice) => ({
    value: voice.name,
    label: `${voice.name} (${voice.lang})`,
  }));

  return (
    <div className={cn('space-y-2', className)}>
      {voices.length > 0 && (
        <Select
          label="Voice"
          value={selectedVoice?.name || ''}
          onChange={(e) => {
            const voice = voices.find((v) => v.name === e.target.value);
            setSelectedVoice(voice || null);
          }}
          options={voiceOptions}
        />
      )}

      <div className="flex gap-2">
        <Button
          onClick={handleSpeak}
          variant={isSpeaking ? 'danger' : 'primary'}
          className="flex items-center gap-2"
          aria-label={isSpeaking ? 'Stop speaking' : 'Start speaking'}
        >
          {isSpeaking ? (
            <>
              <VolumeX className="w-4 h-4" />
              Stop
            </>
          ) : (
            <>
              <Volume2 className="w-4 h-4" />
              Speak
            </>
          )}
        </Button>

        {isSpeaking && (
          <>
            <Button
              onClick={pause}
              variant="secondary"
              className="flex items-center gap-2"
              aria-label="Pause"
            >
              <Pause className="w-4 h-4" />
              Pause
            </Button>
            <Button
              onClick={resume}
              variant="secondary"
              className="flex items-center gap-2"
              aria-label="Resume"
            >
              <Play className="w-4 h-4" />
              Resume
            </Button>
          </>
        )}
      </div>
    </div>
  );
});

TextToSpeech.displayName = 'TextToSpeech';

export default TextToSpeech;



