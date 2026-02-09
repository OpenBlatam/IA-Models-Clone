import type {
  ContinuousAgent,
  CreateAgentRequest,
  UpdateAgentRequest,
  AgentExecutionLog,
} from "../types";
import {
  continuousAgentSchema,
  createAgentRequestSchema,
  updateAgentRequestSchema,
  agentExecutionLogSchema,
} from "../utils/validation/zod-schemas";
import { AgentServerError } from "../utils/errors/agent-errors";
import { httpGet, httpPost, httpPatch, httpDelete } from "./http-client";

const API_BASE = "/api/continuous-agent";

/**
 * Fetches all continuous agents
 * @returns Promise resolving to array of agents
 * @throws AgentError if fetch fails
 */
export const fetchAgents = async (): Promise<ContinuousAgent[]> => {
  return httpGet<ContinuousAgent[]>(
    API_BASE,
    continuousAgentSchema,
    "Error al cargar los agentes"
  );
};

/**
 * Fetches a single agent by ID
 * @param agentId - The ID of the agent to fetch
 * @returns Promise resolving to the agent
 * @throws AgentError if agentId is invalid or fetch fails
 */
export const fetchAgent = async (agentId: string): Promise<ContinuousAgent> => {
  if (!agentId?.trim()) {
    throw new AgentServerError("Agent ID is required");
  }

  return httpGet<ContinuousAgent>(
    `${API_BASE}/${agentId}`,
    continuousAgentSchema,
    "Error al cargar el agente"
  );
};

/**
 * Creates a new continuous agent
 * @param request - The agent creation request
 * @returns Promise resolving to the created agent
 * @throws AgentError if request is invalid or creation fails
 */
export const createAgent = async (
  request: CreateAgentRequest
): Promise<ContinuousAgent> => {
  return httpPost<CreateAgentRequest, ContinuousAgent>(
    API_BASE,
    request,
    createAgentRequestSchema,
    continuousAgentSchema,
    "Error al crear el agente"
  );
};

/**
 * Updates an existing agent
 * @param agentId - The ID of the agent to update
 * @param request - The update request
 * @returns Promise resolving to the updated agent
 * @throws AgentError if agentId is invalid or update fails
 */
export const updateAgent = async (
  agentId: string,
  request: UpdateAgentRequest
): Promise<ContinuousAgent> => {
  if (!agentId?.trim()) {
    throw new AgentServerError("Agent ID is required");
  }

  return httpPatch<UpdateAgentRequest, ContinuousAgent>(
    `${API_BASE}/${agentId}`,
    request,
    updateAgentRequestSchema,
    continuousAgentSchema,
    "Error al actualizar el agente"
  );
};

/**
 * Deletes an agent by ID
 * @param agentId - The ID of the agent to delete
 * @returns Promise resolving when deletion is complete
 * @throws AgentError if agentId is invalid or deletion fails
 */
export const deleteAgent = async (agentId: string): Promise<void> => {
  if (!agentId?.trim()) {
    throw new AgentServerError("Agent ID is required");
  }

  return httpDelete(`${API_BASE}/${agentId}`, "Error al eliminar el agente");
};

/**
 * Fetches execution logs for an agent
 * @param agentId - The ID of the agent
 * @param limit - Maximum number of logs to fetch (default: 50)
 * @returns Promise resolving to array of execution logs
 * @throws AgentError if agentId is invalid or fetch fails
 */
export const fetchAgentLogs = async (
  agentId: string,
  limit: number = 50
): Promise<AgentExecutionLog[]> => {
  if (!agentId?.trim()) {
    throw new AgentServerError("Agent ID is required");
  }
  if (limit < 1 || limit > 1000) {
    throw new AgentServerError("Limit must be between 1 and 1000");
  }

  return httpGet<AgentExecutionLog[]>(
    `${API_BASE}/${agentId}/logs?limit=${limit}`,
    agentExecutionLogSchema,
    "Error al cargar los logs"
  );
};

/**
 * Checks remaining Stripe credits
 * @returns Promise resolving to credits amount or null if unavailable
 */
export const checkStripeCredits = async (): Promise<number | null> => {
  try {
    const response = await fetch(`${API_BASE}/stripe/credits`);
    
    if (!response.ok) {
      return null;
    }

    // Check content type before parsing
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      console.warn("checkStripeCredits: Expected JSON but got:", contentType);
      return null;
    }

    const data = await response.json();
    
    if (typeof data?.credits === "number") {
      return data.credits;
    }
    
    return null;
  } catch (error) {
    if (error instanceof SyntaxError && error.message.includes("JSON")) {
      console.warn("checkStripeCredits: Failed to parse JSON response");
      return null;
    }
    return null;
  }
};

