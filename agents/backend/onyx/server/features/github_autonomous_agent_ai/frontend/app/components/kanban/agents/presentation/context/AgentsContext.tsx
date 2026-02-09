"use client";

import { createContext, useContext, ReactNode } from "react";
import type { AgentEntity } from "../../domain/entities/Agent";

interface AgentsContextValue {
  agents: AgentEntity[];
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

const AgentsContext = createContext<AgentsContextValue | undefined>(undefined);

export function useAgentsContext() {
  const context = useContext(AgentsContext);
  if (!context) {
    throw new Error("useAgentsContext must be used within AgentsProvider");
  }
  return context;
}

interface AgentsProviderProps {
  children: ReactNode;
  value: AgentsContextValue;
}

export function AgentsProvider({ children, value }: AgentsProviderProps) {
  return <AgentsContext.Provider value={value}>{children}</AgentsContext.Provider>;
}








