import React, { useState, useEffect } from 'react';

export const AppVersion: React.FC = () => {
  const [version, setVersion] = useState<string>('');

  useEffect(() => {
    if (typeof window !== 'undefined' && window.electronAPI) {
      window.electronAPI.getVersion().then(setVersion);
    }
  }, []);

  if (!version) return null;

  return (
    <div className="text-xs text-gray-500">
      Versión: {version}
    </div>
  );
};


