/**
 * Dashboard page - Main interface after login.
 * SUNO-style dashboard with sidebar, song creation panel, and song list.
 */

'use client';

import { useState } from 'react';
import { DashboardSidebar } from '@/components/dashboard/sidebar';
import { DashboardTopBar } from '@/components/dashboard/top-bar';
import { SongCreationPanel } from '@/components/dashboard/song-creation-panel';
import { SongListPanel } from '@/components/dashboard/song-list-panel';
import { MusicPlayerBar } from '@/components/dashboard/music-player-bar';

/**
 * Dashboard page component.
 * Main interface displayed after user login.
 *
 * @returns Dashboard page component
 */
export default function DashboardPage() {
  const [activeView, setActiveView] = useState<'simple' | 'custom'>('simple');
  const [selectedSong, setSelectedSong] = useState<string | null>(null);

  return (
    <div className="fixed inset-0 flex flex-col bg-black text-white overflow-hidden">
      {/* Top Bar */}
      <DashboardTopBar activeView={activeView} onViewChange={setActiveView} />

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <DashboardSidebar />

        {/* Center and Right Panels Container */}
        <div className="flex-1 flex overflow-hidden">
          {/* Center Panel - Song Creation */}
          <div className="w-96 border-r border-white/10 overflow-y-auto">
            <SongCreationPanel />
          </div>

          {/* Right Panel - Song List */}
          <div className="flex-1 overflow-y-auto">
            <SongListPanel
              selectedSong={selectedSong}
              onSongSelect={setSelectedSong}
            />
          </div>
        </div>
      </div>

      {/* Bottom Player Bar */}
      <MusicPlayerBar selectedSong={selectedSong} />
    </div>
  );
}

