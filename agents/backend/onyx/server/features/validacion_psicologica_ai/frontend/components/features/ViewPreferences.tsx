/**
 * View preferences component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Toggle } from '@/components/ui';
import { useLocalStorage } from '@/hooks/useLocalStorage';

export interface ViewPreferencesProps {
  className?: string;
}

export const ViewPreferences: React.FC<ViewPreferencesProps> = ({ className }) => {
  const [compactMode, setCompactMode] = useLocalStorage('view-compact-mode', false);
  const [showAnimations, setShowAnimations] = useLocalStorage('view-show-animations', true);
  const [darkMode, setDarkMode] = useLocalStorage('view-dark-mode', false);

  React.useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Preferencias de Vista</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Toggle
          checked={compactMode}
          onChange={setCompactMode}
          label="Modo Compacto"
          id="compact-mode"
        />
        <Toggle
          checked={showAnimations}
          onChange={setShowAnimations}
          label="Mostrar Animaciones"
          id="show-animations"
        />
        <Toggle
          checked={darkMode}
          onChange={setDarkMode}
          label="Modo Oscuro"
          id="dark-mode"
        />
      </CardContent>
    </Card>
  );
};



