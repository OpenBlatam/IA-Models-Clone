/**
 * Recording utilities for Robot 3D View
 * @module robot-3d-view/utils/recording
 */

import type { Position3D, SceneConfig } from '../schemas/validation-schemas';

/**
 * Recording frame
 */
export interface RecordingFrame {
  timestamp: number;
  currentPos: Position3D;
  targetPos: Position3D | null;
  config: SceneConfig;
}

/**
 * Recording data
 */
export interface Recording {
  id: string;
  name: string;
  frames: RecordingFrame[];
  startTime: number;
  endTime: number;
  duration: number;
}

/**
 * Recording Manager class
 */
export class RecordingManager {
  private recordings: Recording[] = [];
  private currentRecording: RecordingFrame[] = [];
  private isRecording = false;
  private startTime = 0;

  /**
   * Starts a new recording
   */
  startRecording(name = `Recording ${Date.now()}`): void {
    if (this.isRecording) {
      this.stopRecording();
    }

    this.isRecording = true;
    this.startTime = Date.now();
    this.currentRecording = [];
  }

  /**
   * Records a frame
   */
  recordFrame(
    currentPos: Position3D,
    targetPos: Position3D | null,
    config: SceneConfig
  ): void {
    if (!this.isRecording) return;

    this.currentRecording.push({
      timestamp: Date.now() - this.startTime,
      currentPos: [...currentPos],
      targetPos: targetPos ? [...targetPos] : null,
      config: { ...config },
    });
  }

  /**
   * Stops recording
   */
  stopRecording(): Recording | null {
    if (!this.isRecording) return null;

    this.isRecording = false;
    const endTime = Date.now();
    const duration = endTime - this.startTime;

    const recording: Recording = {
      id: `recording-${Date.now()}`,
      name: `Recording ${new Date().toISOString()}`,
      frames: [...this.currentRecording],
      startTime: this.startTime,
      endTime,
      duration,
    };

    this.recordings.push(recording);
    this.currentRecording = [];

    return recording;
  }

  /**
   * Gets all recordings
   */
  getRecordings(): readonly Recording[] {
    return [...this.recordings];
  }

  /**
   * Gets a recording by ID
   */
  getRecording(id: string): Recording | undefined {
    return this.recordings.find((r) => r.id === id);
  }

  /**
   * Deletes a recording
   */
  deleteRecording(id: string): boolean {
    const index = this.recordings.findIndex((r) => r.id === id);
    if (index >= 0) {
      this.recordings.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * Exports a recording
   */
  exportRecording(id: string, filename?: string): void {
    const recording = this.getRecording(id);
    if (!recording) return;

    const json = JSON.stringify(recording, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || `${recording.name}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /**
   * Imports a recording
   */
  async importRecording(file: File): Promise<Recording> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (event) => {
        try {
          const text = event.target?.result as string;
          const recording = JSON.parse(text) as Recording;
          this.recordings.push(recording);
          resolve(recording);
        } catch (error) {
          reject(new Error('Invalid recording file'));
        }
      };

      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };

      reader.readAsText(file);
    });
  }

  /**
   * Checks if currently recording
   */
  getIsRecording(): boolean {
    return this.isRecording;
  }
}

/**
 * Global recording manager instance
 */
export const recordingManager = new RecordingManager();



