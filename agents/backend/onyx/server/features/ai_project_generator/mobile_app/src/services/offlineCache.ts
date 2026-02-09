import { storage, STORAGE_KEYS } from '../utils/storage';
import { Project, Stats } from '../types';

const CACHE_DURATION = 5 * 60 * 1000;

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

export const offlineCache = {
  async getProjects(): Promise<Project[] | null> {
    try {
      const cached = await storage.get<CacheEntry<Project[]>>(
        STORAGE_KEYS.CACHED_PROJECTS
      );
      if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        return cached.data;
      }
      return null;
    } catch {
      return null;
    }
  },

  async setProjects(projects: Project[]): Promise<void> {
    try {
      await storage.set<CacheEntry<Project[]>>(STORAGE_KEYS.CACHED_PROJECTS, {
        data: projects,
        timestamp: Date.now(),
      });
    } catch (error) {
      console.error('Error caching projects:', error);
    }
  },

  async getStats(): Promise<Stats | null> {
    try {
      const cached = await storage.get<CacheEntry<Stats>>(
        STORAGE_KEYS.CACHED_STATS
      );
      if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        return cached.data;
      }
      return null;
    } catch {
      return null;
    }
  },

  async setStats(stats: Stats): Promise<void> {
    try {
      await storage.set<CacheEntry<Stats>>(STORAGE_KEYS.CACHED_STATS, {
        data: stats,
        timestamp: Date.now(),
      });
    } catch (error) {
      console.error('Error caching stats:', error);
    }
  },

  async clear(): Promise<void> {
    try {
      await storage.remove(STORAGE_KEYS.CACHED_PROJECTS);
      await storage.remove(STORAGE_KEYS.CACHED_STATS);
      await storage.remove(STORAGE_KEYS.LAST_SYNC);
    } catch (error) {
      console.error('Error clearing cache:', error);
    }
  },
};

