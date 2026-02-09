/**
 * Instructions Overlay Component
 * @module robot-3d-view/controls/instructions-overlay
 */

'use client';

import { memo } from 'react';

/**
 * Instructions Overlay Component
 * 
 * Displays user interaction instructions for the 3D view.
 * 
 * @returns Instructions overlay component
 */
export const InstructionsOverlay = memo(() => {
  return (
    <div className="absolute bottom-4 left-4 bg-gray-800/80 backdrop-blur-sm px-3 py-2 rounded-lg border border-gray-700/50 text-white text-xs">
      <div className="flex items-center gap-4 flex-wrap">
        <span>🖱️ Arrastra para rotar</span>
        <span>🔍 Rueda para zoom</span>
        <span>👆 Click derecho para pan</span>
      </div>
    </div>
  );
});

InstructionsOverlay.displayName = 'InstructionsOverlay';



