import { NextResponse } from "next/server";
import * as Ably from "ably";

export async function GET() {
  return handleTokenRequest();
}

export async function POST() {
  return handleTokenRequest();
}

async function handleTokenRequest() {
  const apiKey = process.env.ABLY_API_KEY;
  
  if (!apiKey) {
    return NextResponse.json(
      { error: "Ably API key is not configured" },
      { status: 500 }
    );
  }

  try {
    const client = new Ably.Rest({
      key: apiKey,
    });

    const tokenRequestData = await client.auth.createTokenRequest({
      clientId: "anonymous",
      capability: {
        "collaboration-invites": ["publish", "subscribe"],
      },
    });

    return NextResponse.json(tokenRequestData);
  } catch (error) {
    return NextResponse.json(
      { error: "Error creating token request" },
      { status: 500 }
    );
  }
}  