import { AgentEntity } from "../../domain/entities/Agent";
import type { IAgentRepository } from "../../domain/repositories/IAgentRepository";
import { API_ENDPOINTS } from "../../config/constants";
import { AgentAdapter } from "../adapters/AgentAdapter";
import { AgentNotFoundError, AgentOperationError } from "../../domain/errors/AgentErrors";
import { Result } from "../../utils/result";

export class AgentRepository implements IAgentRepository {
  constructor(private readonly adapter: AgentAdapter) {}

  async findAll(): Promise<Result<AgentEntity[], Error>> {
    try {
      const res = await fetch(API_ENDPOINTS.AGENTS);
      if (!res.ok) {
        // Check if response is HTML (error page)
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("text/html")) {
          const text = await res.text();
          return Result.err(
            new AgentOperationError(
              `API returned HTML instead of JSON. Status: ${res.status} ${res.statusText}. This usually means the endpoint doesn't exist or the server is returning an error page.`,
              "findAll"
            )
          );
        }
        return Result.err(
          new AgentOperationError(
            `Failed to fetch agents: ${res.status} ${res.statusText}`,
            "findAll"
          )
        );
      }
      
      // Check content type before parsing
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        const text = await res.text();
        console.error("Expected JSON but got:", contentType, text.substring(0, 200));
        return Result.err(
          new AgentOperationError(
            `API returned non-JSON response. Content-Type: ${contentType}`,
            "findAll"
          )
        );
      }
      
      const data = await res.json();
      const agents = Array.isArray(data) ? data : [];
      const entities = agents.map((agent) => this.adapter.toDomain(agent));
      return Result.ok(entities);
    } catch (error) {
      console.error("Error fetching agents", error);
      
      // Handle network errors (backend not available)
      if (
        error instanceof TypeError ||
        (error instanceof Error && (
          error.message.includes("fetch") ||
          error.message.includes("Failed to fetch") ||
          error.message.includes("NetworkError") ||
          error.message.includes("network") ||
          error.message.includes("ECONNREFUSED") ||
          error.message.includes("ENOTFOUND")
        ))
      ) {
        return Result.err(
          new AgentOperationError(
            "Cannot connect to the backend server. Please ensure the backend is running and accessible.",
            "findAll"
          )
        );
      }
      
      if (error instanceof SyntaxError && error.message.includes("JSON")) {
        return Result.err(
          new AgentOperationError(
            "Failed to parse JSON response. The API may have returned HTML or invalid JSON.",
            "findAll"
          )
        );
      }
      return Result.err(
        error instanceof Error
          ? error
          : new AgentOperationError("Unknown error fetching agents", "findAll")
      );
    }
  }

  async findById(id: string): Promise<Result<AgentEntity | null, Error>> {
    try {
      const res = await fetch(API_ENDPOINTS.AGENT_BY_ID(id));
      if (!res.ok) {
        if (res.status === 404) {
          return Result.ok(null);
        }
        // Check if response is HTML (error page)
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("text/html")) {
          return Result.err(
            new AgentOperationError(
              `API returned HTML instead of JSON. Status: ${res.status} ${res.statusText}. This usually means the endpoint doesn't exist or the server is returning an error page.`,
              "findById"
            )
          );
        }
        return Result.err(
          new AgentNotFoundError(id)
        );
      }
      
      // Check content type before parsing
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        const text = await res.text();
        console.error("Expected JSON but got:", contentType, text.substring(0, 200));
        return Result.err(
          new AgentOperationError(
            `API returned non-JSON response. Content-Type: ${contentType}`,
            "findById"
          )
        );
      }
      
      const data = await res.json();
      const entity = this.adapter.toDomain(data);
      return Result.ok(entity);
    } catch (error) {
      console.error("Error fetching agent", error);
      
      // Handle network errors (backend not available)
      if (
        error instanceof TypeError ||
        (error instanceof Error && (
          error.message.includes("fetch") ||
          error.message.includes("Failed to fetch") ||
          error.message.includes("NetworkError") ||
          error.message.includes("network") ||
          error.message.includes("ECONNREFUSED") ||
          error.message.includes("ENOTFOUND")
        ))
      ) {
        return Result.err(
          new AgentOperationError(
            "Cannot connect to the backend server. Please ensure the backend is running and accessible.",
            "findById"
          )
        );
      }
      
      if (error instanceof SyntaxError && error.message.includes("JSON")) {
        return Result.err(
          new AgentOperationError(
            "Failed to parse JSON response. The API may have returned HTML or invalid JSON.",
            "findById"
          )
        );
      }
      return Result.err(
        error instanceof Error
          ? error
          : new AgentOperationError("Unknown error fetching agent", "findById")
      );
    }
  }

  async toggleActive(id: string, isActive: boolean): Promise<Result<AgentEntity, Error>> {
    try {
      const res = await fetch(API_ENDPOINTS.AGENT_BY_ID(id), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ isActive }),
      });

      if (!res.ok) {
        // Check if response is HTML (error page)
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("text/html")) {
          const text = await res.text();
          return Result.err(
            new AgentOperationError(
              `API returned HTML instead of JSON. Status: ${res.status} ${res.statusText}. This usually means the endpoint doesn't exist or the server is returning an error page.`,
              "toggleActive"
            )
          );
        }
        return Result.err(
          new AgentOperationError(
            `Failed to toggle agent: ${res.status} ${res.statusText}`,
            "toggleActive"
          )
        );
      }

      // Check content type before parsing
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        const text = await res.text();
        console.error("Expected JSON but got:", contentType, text.substring(0, 200));
        return Result.err(
          new AgentOperationError(
            `API returned non-JSON response. Content-Type: ${contentType}`,
            "toggleActive"
          )
        );
      }

      const data = await res.json();
      const entity = this.adapter.toDomain(data);
      return Result.ok(entity);
    } catch (error) {
      console.error("Error toggling agent", error);
      
      // Handle network errors (backend not available)
      if (
        error instanceof TypeError ||
        (error instanceof Error && (
          error.message.includes("fetch") ||
          error.message.includes("Failed to fetch") ||
          error.message.includes("NetworkError") ||
          error.message.includes("network") ||
          error.message.includes("ECONNREFUSED") ||
          error.message.includes("ENOTFOUND")
        ))
      ) {
        return Result.err(
          new AgentOperationError(
            "Cannot connect to the backend server. Please ensure the backend is running and accessible.",
            "toggleActive"
          )
        );
      }
      
      if (error instanceof SyntaxError && error.message.includes("JSON")) {
        return Result.err(
          new AgentOperationError(
            "Failed to parse JSON response. The API may have returned HTML or invalid JSON.",
            "toggleActive"
          )
        );
      }
      return Result.err(
        error instanceof Error
          ? error
          : new AgentOperationError("Unknown error toggling agent", "toggleActive")
      );
    }
  }
}
