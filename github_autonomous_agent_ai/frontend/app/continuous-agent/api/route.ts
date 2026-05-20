import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/continuous-agent`, {
      headers: {
        Cookie: request.headers.get("cookie") || "",
      },
    });

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
          { detail: error.detail || "Failed to fetch agents" },
          { status: response.status }
        );
      } catch {
        return NextResponse.json(
          { detail: `Failed to fetch agents: ${response.statusText}` },
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
    console.error("Error fetching agents:", error);
    
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

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const response = await fetch(`${BACKEND_URL}/api/continuous-agent`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Cookie: request.headers.get("cookie") || "",
      },
      body: JSON.stringify(body),
    });

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
          { detail: error.detail || "Failed to create agent" },
          { status: response.status }
        );
      } catch {
        return NextResponse.json(
          { detail: `Failed to create agent: ${response.statusText}` },
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
    console.error("Error creating agent:", error);
    
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







