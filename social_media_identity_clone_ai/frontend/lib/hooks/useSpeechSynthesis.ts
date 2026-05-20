import { useState, useCallback, useEffect } from 'react';

interface UseSpeechSynthesisOptions {
  pitch?: number;
  rate?: number;
  volume?: number;
  lang?: string;
  voice?: SpeechSynthesisVoice;
  onEnd?: () => void;
  onError?: (event: SpeechSynthesisErrorEvent) => void;
}

export const useSpeechSynthesis = (options: UseSpeechSynthesisOptions = {}) => {
  const {
    pitch = 1,
    rate = 1,
    volume = 1,
    lang = 'en-US',
    voice,
    onEnd,
    onError,
  } = options;

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return;
    }

    const loadVoices = (): void => {
      const availableVoices = window.speechSynthesis.getVoices();
      setVoices(availableVoices);
    };

    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;

    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  const speak = useCallback(
    (text: string) => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        return;
      }

      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.pitch = pitch;
      utterance.rate = rate;
      utterance.volume = volume;
      utterance.lang = lang;

      if (voice) {
        utterance.voice = voice;
      }

      utterance.onstart = () => {
        setIsSpeaking(true);
      };

      utterance.onend = () => {
        setIsSpeaking(false);
        if (onEnd) {
          onEnd();
        }
      };

      utterance.onerror = (event) => {
        setIsSpeaking(false);
        if (onError) {
          onError(event);
        }
      };

      window.speechSynthesis.speak(utterance);
    },
    [pitch, rate, volume, lang, voice, onEnd, onError]
  );

  const cancel = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  }, []);

  const pause = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.pause();
    }
  }, []);

  const resume = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.resume();
    }
  }, []);

  return {
    isSpeaking,
    speak,
    cancel,
    pause,
    resume,
    voices,
    supported: typeof window !== 'undefined' && 'speechSynthesis' in window,
  };
};



