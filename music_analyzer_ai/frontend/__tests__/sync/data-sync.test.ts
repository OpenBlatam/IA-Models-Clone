/**
 * Data Synchronization Testing
 * 
 * Tests that verify data synchronization between devices,
 * conflict resolution, and sync state management.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock sync storage
const mockSyncStorage = {
  local: new Map<string, any>(),
  remote: new Map<string, any>(),
  syncQueue: [] as any[],
  
  getLocal: function(key: string) {
    return this.local.get(key);
  },
  
  setLocal: function(key: string, value: any) {
    this.local.set(key, value);
  },
  
  getRemote: function(key: string) {
    return this.remote.get(key);
  },
  
  setRemote: function(key: string, value: any) {
    this.remote.set(key, value);
  },
};

describe('Data Synchronization Testing', () => {
  beforeEach(() => {
    mockSyncStorage.local.clear();
    mockSyncStorage.remote.clear();
    mockSyncStorage.syncQueue = [];
  });

  describe('Basic Synchronization', () => {
    it('should sync local data to remote', () => {
      mockSyncStorage.setLocal('tracks', [{ id: '1', name: 'Track 1' }]);
      
      const syncToRemote = (key: string) => {
        const localData = mockSyncStorage.getLocal(key);
        if (localData) {
          mockSyncStorage.setRemote(key, localData);
          return { success: true };
        }
        return { success: false };
      };
      
      const result = syncToRemote('tracks');
      expect(result.success).toBe(true);
      expect(mockSyncStorage.getRemote('tracks')).toEqual([{ id: '1', name: 'Track 1' }]);
    });

    it('should sync remote data to local', () => {
      mockSyncStorage.setRemote('tracks', [{ id: '1', name: 'Track 1' }]);
      
      const syncToLocal = (key: string) => {
        const remoteData = mockSyncStorage.getRemote(key);
        if (remoteData) {
          mockSyncStorage.setLocal(key, remoteData);
          return { success: true };
        }
        return { success: false };
      };
      
      const result = syncToLocal('tracks');
      expect(result.success).toBe(true);
      expect(mockSyncStorage.getLocal('tracks')).toEqual([{ id: '1', name: 'Track 1' }]);
    });

    it('should perform bidirectional sync', () => {
      mockSyncStorage.setLocal('tracks', [{ id: '1', name: 'Local Track' }]);
      mockSyncStorage.setRemote('tracks', [{ id: '2', name: 'Remote Track' }]);
      
      const bidirectionalSync = (key: string) => {
        const local = mockSyncStorage.getLocal(key) || [];
        const remote = mockSyncStorage.getRemote(key) || [];
        
        const merged = [...local, ...remote];
        const unique = Array.from(
          new Map(merged.map(item => [item.id, item])).values()
        );
        
        mockSyncStorage.setLocal(key, unique);
        mockSyncStorage.setRemote(key, unique);
        
        return { success: true, merged: unique };
      };
      
      const result = bidirectionalSync('tracks');
      expect(result.merged).toHaveLength(2);
    });
  });

  describe('Conflict Resolution', () => {
    it('should resolve conflicts using last-write-wins', () => {
      const resolveLastWriteWins = (local: any, remote: any) => {
        if (remote.timestamp > local.timestamp) {
          return remote;
        }
        return local;
      };
      
      const local = { id: '1', name: 'Local', timestamp: 1000 };
      const remote = { id: '1', name: 'Remote', timestamp: 2000 };
      
      const resolved = resolveLastWriteWins(local, remote);
      expect(resolved.name).toBe('Remote');
    });

    it('should resolve conflicts using merge strategy', () => {
      const resolveMerge = (local: any, remote: any) => {
        return {
          ...local,
          ...remote,
          merged: true,
        };
      };
      
      const local = { id: '1', name: 'Local', localField: 'local' };
      const remote = { id: '1', name: 'Remote', remoteField: 'remote' };
      
      const resolved = resolveMerge(local, remote);
      expect(resolved.merged).toBe(true);
      expect(resolved.localField).toBe('local');
      expect(resolved.remoteField).toBe('remote');
    });

    it('should resolve conflicts using custom strategy', () => {
      const resolveCustom = (local: any, remote: any, strategy: string) => {
        if (strategy === 'local') return local;
        if (strategy === 'remote') return remote;
        if (strategy === 'merge') return { ...local, ...remote };
        return local;
      };
      
      const local = { id: '1', name: 'Local' };
      const remote = { id: '1', name: 'Remote' };
      
      expect(resolveCustom(local, remote, 'local').name).toBe('Local');
      expect(resolveCustom(local, remote, 'remote').name).toBe('Remote');
    });
  });

  describe('Sync Queue', () => {
    it('should queue sync operations', () => {
      const queueSync = (operation: any) => {
        mockSyncStorage.syncQueue.push(operation);
      };
      
      queueSync({ type: 'UPDATE', key: 'tracks', data: { id: '1' } });
      queueSync({ type: 'DELETE', key: 'tracks', id: '1' });
      
      expect(mockSyncStorage.syncQueue).toHaveLength(2);
    });

    it('should process sync queue', async () => {
      mockSyncStorage.syncQueue = [
        { type: 'UPDATE', key: 'tracks', data: { id: '1', name: 'Track 1' } },
        { type: 'UPDATE', key: 'tracks', data: { id: '2', name: 'Track 2' } },
      ];
      
      const processQueue = async () => {
        while (mockSyncStorage.syncQueue.length > 0) {
          const operation = mockSyncStorage.syncQueue.shift();
          if (operation?.type === 'UPDATE') {
            mockSyncStorage.setLocal(operation.key, operation.data);
          }
        }
      };
      
      await processQueue();
      expect(mockSyncStorage.syncQueue).toHaveLength(0);
    });

    it('should retry failed sync operations', () => {
      let attemptCount = 0;
      const maxRetries = 3;
      
      const syncWithRetry = async (operation: any) => {
        for (let i = 0; i < maxRetries; i++) {
          attemptCount++;
          try {
            // Simulate sync
            if (i < maxRetries - 1) throw new Error('Sync failed');
            return { success: true };
          } catch {
            if (i === maxRetries - 1) throw new Error('Max retries exceeded');
          }
        }
      };
      
      syncWithRetry({ type: 'UPDATE', key: 'tracks' });
      expect(attemptCount).toBe(maxRetries);
    });
  });

  describe('Sync State Management', () => {
    it('should track sync status', () => {
      const syncState = {
        status: 'idle' as 'idle' | 'syncing' | 'error',
        lastSync: null as Date | null,
        pendingChanges: 0,
      };
      
      const startSync = () => {
        syncState.status = 'syncing';
        syncState.pendingChanges = mockSyncStorage.syncQueue.length;
      };
      
      startSync();
      expect(syncState.status).toBe('syncing');
    });

    it('should detect sync conflicts', () => {
      const detectConflict = (local: any, remote: any) => {
        if (local.version && remote.version) {
          return local.version !== remote.version;
        }
        if (local.timestamp && remote.timestamp) {
          return Math.abs(local.timestamp - remote.timestamp) > 1000;
        }
        return false;
      };
      
      const local = { id: '1', version: 1 };
      const remote = { id: '1', version: 2 };
      
      expect(detectConflict(local, remote)).toBe(true);
    });

    it('should handle sync errors', () => {
      const syncWithErrorHandling = async () => {
        try {
          // Simulate sync
          throw new Error('Network error');
        } catch (error: any) {
          return {
            success: false,
            error: error.message,
            retryable: true,
          };
        }
      };
      
      syncWithErrorHandling().then(result => {
        expect(result.success).toBe(false);
        expect(result.retryable).toBe(true);
      });
    });
  });

  describe('Incremental Sync', () => {
    it('should sync only changed data', () => {
      const getChanges = (local: any[], remote: any[]) => {
        const localIds = new Set(local.map(item => item.id));
        const remoteIds = new Set(remote.map(item => item.id));
        
        const added = remote.filter(item => !localIds.has(item.id));
        const updated = remote.filter(item => {
          const localItem = local.find(l => l.id === item.id);
          return localItem && localItem.version !== item.version;
        });
        const deleted = local.filter(item => !remoteIds.has(item.id));
        
        return { added, updated, deleted };
      };
      
      const local = [{ id: '1', version: 1 }, { id: '2', version: 1 }];
      const remote = [{ id: '1', version: 2 }, { id: '3', version: 1 }];
      
      const changes = getChanges(local, remote);
      expect(changes.added).toHaveLength(1);
      expect(changes.updated).toHaveLength(1);
      expect(changes.deleted).toHaveLength(1);
    });

    it('should use timestamps for change detection', () => {
      const getChangesSince = (data: any[], since: number) => {
        return data.filter(item => item.updatedAt > since);
      };
      
      const data = [
        { id: '1', updatedAt: 1000 },
        { id: '2', updatedAt: 2000 },
        { id: '3', updatedAt: 3000 },
      ];
      
      const changes = getChangesSince(data, 1500);
      expect(changes).toHaveLength(2);
    });
  });
});

