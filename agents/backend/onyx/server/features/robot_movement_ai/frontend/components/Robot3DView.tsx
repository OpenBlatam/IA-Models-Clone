/**
 * Robot3DView - Legacy Export
 * 
 * This file maintains backward compatibility while using the refactored modular component.
 * The new implementation is located in components/robot-3d-view/
 * 
 * @deprecated Use components/robot-3d-view/index.tsx directly for better tree-shaking
 * @module components/Robot3DView
 */

'use client';

// Re-export the refactored component for backward compatibility
export { default } from './robot-3d-view';
export type { Robot3DViewProps } from './robot-3d-view';
