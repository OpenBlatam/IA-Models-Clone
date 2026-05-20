"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

interface QueryContextType {
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

const QueryContext = createContext<QueryContextType>({
  isLoading: false,
  error: null,
  refetch: () => {}
});

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refetch = () => {
    setIsLoading(true);
    setError(null);
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  };

  return (
    <QueryContext.Provider value={{ isLoading, error, refetch }}>
      {children}
    </QueryContext.Provider>
  );
}

export const useQuery = () => useContext(QueryContext);
