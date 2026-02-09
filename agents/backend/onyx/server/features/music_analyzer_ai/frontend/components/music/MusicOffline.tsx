'use client';

import { useState, useEffect } from 'react';
import { Wifi, WifiOff, Download, CheckCircle } from 'lucide-react';

export function MusicOffline() {
  const [isOnline, setIsOnline] = useState(true);
  const [offlineData, setOfflineData] = useState<any[]>([]);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    setIsOnline(navigator.onLine);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Cargar datos offline desde localStorage
    const saved = localStorage.getItem('offline-music-data');
    if (saved) {
      try {
        setOfflineData(JSON.parse(saved));
      } catch (e) {
        console.error('Error loading offline data:', e);
      }
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (isOnline) {
    return null;
  }

  return (
    <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 bg-yellow-500/90 backdrop-blur-lg rounded-xl p-4 border border-yellow-500 shadow-lg">
      <div className="flex items-center gap-3">
        <WifiOff className="w-5 h-5 text-white" />
        <div>
          <p className="text-white font-medium">Modo Offline</p>
          <p className="text-white/80 text-sm">
            {offlineData.length} elemento{offlineData.length !== 1 ? 's' : ''} disponible{offlineData.length !== 1 ? 's' : ''} offline
          </p>
        </div>
      </div>
    </div>
  );
}


