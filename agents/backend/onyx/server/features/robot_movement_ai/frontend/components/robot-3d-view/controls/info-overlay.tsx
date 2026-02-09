/**
 * Info Overlay Component
 * @module robot-3d-view/controls/info-overlay
 */

'use client';

import { memo } from 'react';
import type { Position3D } from '../types';
import type { RobotStatus as StoreRobotStatus } from '@/lib/api/types';

/**
 * Props for InfoOverlay component
 */
interface InfoOverlayProps {
  currentPos: Position3D;
  targetPos: Position3D | null;
  status: StoreRobotStatus | null;
}

/**
 * Info Overlay Component
 * 
 * Displays robot status information overlay with current position, target, and connection status.
 * 
 * @param props - Position and status data
 * @returns Info overlay component
 */
export const InfoOverlay = memo(({ currentPos, targetPos, status }: InfoOverlayProps) => {
  const isConnected = status?.robot_status?.connected ?? false;

  return (
    <div className="absolute top-4 left-4 bg-gray-800/95 backdrop-blur-md p-4 rounded-lg border border-gray-700/50 text-white text-sm shadow-xl">
      <div className="space-y-2">
        <div className="flex items-center gap-2 mb-2">
          <div className={`w-2 h-2 rounded-full animate-pulse ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="font-semibold text-xs uppercase tracking-wide">Estado del Robot</span>
        </div>
        <div>
          <span className="text-gray-400 text-xs">Posición Actual: </span>
          <span className="font-mono text-sm font-semibold">
            ({currentPos[0].toFixed(2)}, {currentPos[1].toFixed(2)}, {currentPos[2].toFixed(2)})
          </span>
        </div>
        {targetPos && (
          <div>
            <span className="text-gray-400 text-xs">Objetivo: </span>
            <span className="font-mono text-sm font-semibold text-yellow-400">
              ({targetPos[0].toFixed(2)}, {targetPos[1].toFixed(2)}, {targetPos[2].toFixed(2)})
            </span>
          </div>
        )}
        {status && (
          <div className="pt-2 border-t border-gray-700">
            <span className="text-gray-400 text-xs">Estado: </span>
            <span className={`text-xs font-semibold ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              {isConnected ? 'Conectado' : 'Desconectado'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
});

InfoOverlay.displayName = 'InfoOverlay';



