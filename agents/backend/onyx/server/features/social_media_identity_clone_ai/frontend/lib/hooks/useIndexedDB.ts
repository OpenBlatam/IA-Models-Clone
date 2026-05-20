import { useState, useEffect, useCallback, useRef } from 'react';

interface UseIndexedDBOptions {
  dbName: string;
  version?: number;
  stores: Array<{ name: string; keyPath?: string; autoIncrement?: boolean }>;
}

export const useIndexedDB = (options: UseIndexedDBOptions) => {
  const { dbName, version = 1, stores } = options;
  const [db, setDb] = useState<IDBDatabase | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const dbRef = useRef<IDBDatabase | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !('indexedDB' in window)) {
      setError(new Error('IndexedDB is not supported'));
      return;
    }

    const request = indexedDB.open(dbName, version);

    request.onerror = () => {
      setError(new Error('Failed to open IndexedDB'));
    };

    request.onsuccess = () => {
      const database = request.result;
      setDb(database);
      dbRef.current = database;
    };

    request.onupgradeneeded = (event) => {
      const database = (event.target as IDBOpenDBRequest).result;
      stores.forEach((store) => {
        if (!database.objectStoreNames.contains(store.name)) {
          database.createObjectStore(store.name, {
            keyPath: store.keyPath,
            autoIncrement: store.autoIncrement,
          });
        }
      });
    };

    return () => {
      if (dbRef.current) {
        dbRef.current.close();
      }
    };
  }, [dbName, version, stores]);

  const add = useCallback(
    <T,>(storeName: string, data: T): Promise<IDBValidKey> => {
      return new Promise((resolve, reject) => {
        if (!dbRef.current) {
          reject(new Error('Database not initialized'));
          return;
        }

        const transaction = dbRef.current.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.add(data);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    },
    []
  );

  const get = useCallback(
    <T,>(storeName: string, key: IDBValidKey): Promise<T | undefined> => {
      return new Promise((resolve, reject) => {
        if (!dbRef.current) {
          reject(new Error('Database not initialized'));
          return;
        }

        const transaction = dbRef.current.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.get(key);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    },
    []
  );

  const getAll = useCallback(
    <T,>(storeName: string): Promise<T[]> => {
      return new Promise((resolve, reject) => {
        if (!dbRef.current) {
          reject(new Error('Database not initialized'));
          return;
        }

        const transaction = dbRef.current.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.getAll();

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    },
    []
  );

  const put = useCallback(
    <T,>(storeName: string, data: T): Promise<IDBValidKey> => {
      return new Promise((resolve, reject) => {
        if (!dbRef.current) {
          reject(new Error('Database not initialized'));
          return;
        }

        const transaction = dbRef.current.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.put(data);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    },
    []
  );

  const remove = useCallback(
    (storeName: string, key: IDBValidKey): Promise<void> => {
      return new Promise((resolve, reject) => {
        if (!dbRef.current) {
          reject(new Error('Database not initialized'));
          return;
        }

        const transaction = dbRef.current.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.delete(key);

        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    },
    []
  );

  return {
    db,
    error,
    add,
    get,
    getAll,
    put,
    remove,
    supported: typeof window !== 'undefined' && 'indexedDB' in window,
  };
};



