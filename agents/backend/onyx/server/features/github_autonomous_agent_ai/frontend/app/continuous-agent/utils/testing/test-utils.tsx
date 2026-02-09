/**
 * Testing utilities for Continuous Agent components
 * 
 * Provides helper functions and utilities for testing React components
 */

import React, { type ReactElement } from "react";
import { render, type RenderOptions } from "@testing-library/react";
import type { ContinuousAgent, CreateAgentRequest } from "../../types";

/**
 * Test data factory for creating mock agents
 */
export const createMockAgent = (overrides?: Partial<ContinuousAgent>): ContinuousAgent => {
  const defaultAgent: ContinuousAgent = {
    id: `agent-${Math.random().toString(36).substr(2, 9)}`,
    name: "Test Agent",
    description: "Test agent description",
    isActive: true,
    config: {
      taskType: "content_generation",
      frequency: 3600,
      parameters: {},
    },
    stats: {
      totalExecutions: 0,
      successfulExecutions: 0,
      failedExecutions: 0,
      lastExecutionAt: null,
      nextExecutionAt: null,
      creditsUsed: 0,
      averageExecutionTime: 0,
    },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    stripeCreditsRemaining: 1000,
    ...overrides,
  };

  return defaultAgent;
};

/**
 * Test data factory for creating mock agent requests
 */
export const createMockAgentRequest = (
  overrides?: Partial<CreateAgentRequest>
): CreateAgentRequest => {
  return {
    name: "Test Agent",
    description: "Test agent description",
    config: {
      taskType: "content_generation",
      frequency: 3600,
      parameters: {},
    },
    ...overrides,
  };
};

/**
 * Creates an array of mock agents
 */
export const createMockAgents = (count: number): ContinuousAgent[] => {
  return Array.from({ length: count }, (_, index) =>
    createMockAgent({
      id: `agent-${index}`,
      name: `Test Agent ${index + 1}`,
    })
  );
};

/**
 * Custom render function with providers
 */
interface CustomRenderOptions extends Omit<RenderOptions, "wrapper"> {
  // Add custom options here
}

/**
 * Custom render function that includes all providers
 */
export function renderWithProviders(
  ui: ReactElement,
  options: CustomRenderOptions = {}
): ReturnType<typeof render> {
  // Add any global providers here (e.g., ThemeProvider, QueryClientProvider)
  const Wrapper = ({ children }: { children: React.ReactNode }) => {
    return <>{children}</>;
  };

  return render(ui, { wrapper: Wrapper, ...options });
}

/**
 * Waits for async operations to complete
 */
export async function waitForAsync(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Mocks fetch with a response
 */
export function mockFetchResponse(data: unknown, status: number = 200): void {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: status >= 200 && status < 300,
      status,
      json: () => Promise.resolve(data),
      text: () => Promise.resolve(JSON.stringify(data)),
    } as Response)
  );
}

/**
 * Mocks fetch with an error
 */
export function mockFetchError(error: Error): void {
  global.fetch = jest.fn(() => Promise.reject(error));
}

/**
 * Creates a mock error
 */
export function createMockError(message: string, code?: string): Error {
  const error = new Error(message);
  if (code) {
    (error as Error & { code: string }).code = code;
  }
  return error;
}

/**
 * Test utilities re-export
 */
export * from "@testing-library/react";
export { default as userEvent } from "@testing-library/user-event";




