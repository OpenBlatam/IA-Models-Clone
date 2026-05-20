import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

export async function GET(
  request: NextRequest,
  { params }: { params: { agentId: string } }
) {
  try {
    const response = await fetch(
      `${BACKEND_URL}/api/continuous-agent/${params.agentId}`,
      {
        headers: {
          Cookie: request.headers.get("cookie") || "",
        },
      }
    );

    if (!response.ok) {
      // Check if response is HTML (error page)
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("text/html")) {
        // Only log in development
        if (process.env.NODE_ENV === "development") {
          const text = await response.text();
          console.warn("Backend returned HTML instead of JSON:", {
            status: response.status,
            statusText: response.statusText,
            preview: text.substring(0, 200),
          });
        }
        return NextResponse.json(
          { detail: `Backend server error. Status: ${response.status} ${response.statusText}. Please check if the backend is running.` },
          { status: 502 }
        );
      }
      
      // Try to parse error as JSON
      try {
        const error = await response.json();
        return NextResponse.json(
          { detail: error.detail || "Failed to fetch agent" },
          { status: response.status }
        );
      } catch {
        return NextResponse.json(
          { detail: `Failed to fetch agent: ${response.statusText}` },
          { status: response.status }
        );
      }
    }

    // Check content type before parsing
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      // Only log in development
      if (process.env.NODE_ENV === "development") {
        const text = await response.text();
        console.warn("Backend returned non-JSON:", {
          contentType,
          preview: text.substring(0, 200),
        });
      }
      return NextResponse.json(
        { detail: `Backend returned non-JSON response. Content-Type: ${contentType}` },
        { status: 502 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching agent:", error);
    
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
      return NextResponse.json(
        { 
          detail: `Cannot connect to backend server at ${BACKEND_URL}. Please ensure the backend is running and accessible.` 
        },
        { status: 503 }
      );
    }
    
    if (error instanceof SyntaxError && error.message.includes("JSON")) {
      return NextResponse.json(
        { detail: "Failed to parse JSON response from backend" },
        { status: 502 }
      );
    }
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Internal server error" },
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { agentId: string } }
) {
  try {
    const body = await request.json();

    const response = await fetch(
      `${BACKEND_URL}/api/continuous-agent/${params.agentId}`,
      {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Cookie: request.headers.get("cookie") || "",
        },
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      // Check if response is HTML (error page)
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("text/html")) {
        // Only log in development
        if (process.env.NODE_ENV === "development") {
          const text = await response.text();
          console.warn("Backend returned HTML instead of JSON:", {
            status: response.status,
            statusText: response.statusText,
            preview: text.substring(0, 200),
          });
        }
        return NextResponse.json(
          { detail: `Backend server error. Status: ${response.status} ${response.statusText}. Please check if the backend is running.` },
          { status: 502 }
        );
      }
      
      // Try to parse error as JSON
      try {
        const error = await response.json();
        return NextResponse.json(
          { detail: error.detail || "Failed to update agent" },
          { status: response.status }
        );
      } catch {
        return NextResponse.json(
          { detail: `Failed to update agent: ${response.statusText}` },
          { status: response.status }
        );
      }
    }

    // Check content type before parsing
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      // Only log in development
      if (process.env.NODE_ENV === "development") {
        const text = await response.text();
        console.warn("Backend returned non-JSON:", {
          contentType,
          preview: text.substring(0, 200),
        });
      }
      return NextResponse.json(
        { detail: `Backend returned non-JSON response. Content-Type: ${contentType}` },
        { status: 502 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error updating agent:", error);
    
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
      return NextResponse.json(
        { 
          detail: `Cannot connect to backend server at ${BACKEND_URL}. Please ensure the backend is running and accessible.` 
        },
        { status: 503 }
      );
    }
    
    if (error instanceof SyntaxError && error.message.includes("JSON")) {
      return NextResponse.json(
        { detail: "Failed to parse JSON response from backend" },
        { status: 502 }
      );
    }
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Internal server error" },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { agentId: string } }
) {
  try {
    const response = await fetch(
      `${BACKEND_URL}/api/continuous-agent/${params.agentId}`,
      {
        method: "DELETE",
        headers: {
          Cookie: request.headers.get("cookie") || "",
        },
      }
    );

    if (!response.ok) {
      // Check if response is HTML (error page)
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("text/html")) {
        // Only log in development
        if (process.env.NODE_ENV === "development") {
          const text = await response.text();
          console.warn("Backend returned HTML instead of JSON:", {
            status: response.status,
            statusText: response.statusText,
            preview: text.substring(0, 200),
          });
        }
        return NextResponse.json(
          { detail: `Backend server error. Status: ${response.status} ${response.statusText}. Please check if the backend is running.` },
          { status: 502 }
        );
      }
      
      // Try to parse error as JSON
      try {
        const error = await response.json();
        return NextResponse.json(
          { detail: error.detail || "Failed to delete agent" },
          { status: response.status }
        );
      } catch {
        return NextResponse.json(
          { detail: `Failed to delete agent: ${response.statusText}` },
          { status: response.status }
        );
      }
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting agent:", error);
    
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
      return NextResponse.json(
        { 
          detail: `Cannot connect to backend server at ${BACKEND_URL}. Please ensure the backend is running and accessible.` 
        },
        { status: 503 }
      );
    }
    
    if (error instanceof SyntaxError && error.message.includes("JSON")) {
      return NextResponse.json(
        { detail: "Failed to parse JSON response from backend" },
        { status: 502 }
      );
    }
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Internal server error" },
      { status: 500 }
    );
  }
}







