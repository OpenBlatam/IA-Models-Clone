import { useEffect, useState } from 'react';
import { InteractionManager } from 'react-native';

export function useInteractionReady(): boolean {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const interaction = InteractionManager.runAfterInteractions(() => {
      setReady(true);
    });

    return () => {
      interaction.cancel();
    };
  }, []);

  return ready;
}

