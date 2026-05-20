import { create } from 'zustand';
import { Position } from '../api/types';

interface MovementRecord {
  timestamp: number;
  position: Position;
  action: 'move' | 'stop' | 'home';
}

interface RecordingState {
  isRecording: boolean;
  isPlaying: boolean;
  records: MovementRecord[];
  currentIndex: number;
  startRecording: () => void;
  stopRecording: () => void;
  addRecord: (position: Position, action: 'move' | 'stop' | 'home') => void;
  clearRecords: () => void;
  startPlayback: () => void;
  stopPlayback: () => void;
  nextRecord: () => MovementRecord | null;
}

export const useRecordingStore = create<RecordingState>((set, get) => ({
  isRecording: false,
  isPlaying: false,
  records: [],
  currentIndex: 0,

  startRecording: () => {
    set({ isRecording: true, records: [], currentIndex: 0 });
  },

  stopRecording: () => {
    set({ isRecording: false });
  },

  addRecord: (position, action) => {
    if (get().isRecording) {
      set((state) => ({
        records: [
          ...state.records,
          {
            timestamp: Date.now(),
            position,
            action,
          },
        ],
      }));
    }
  },

  clearRecords: () => {
    set({ records: [], currentIndex: 0 });
  },

  startPlayback: () => {
    set({ isPlaying: true, currentIndex: 0 });
  },

  stopPlayback: () => {
    set({ isPlaying: false, currentIndex: 0 });
  },

  nextRecord: () => {
    const { records, currentIndex } = get();
    if (currentIndex < records.length) {
      set({ currentIndex: currentIndex + 1 });
      return records[currentIndex];
    }
    set({ isPlaying: false, currentIndex: 0 });
    return null;
  },
}));

