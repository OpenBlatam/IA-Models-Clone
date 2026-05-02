import { NextRequest, NextResponse } from "next/server";
import { openai } from "@/lib/openai";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  try {
    // Check if API key is configured
    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json(
        { 
          error: "OpenAI API key not configured",
          details: "Please add OPENAI_API_KEY to your .env.local file"
        },
        { status: 500 }
      );
    }

    // Test OpenAI connection with a simple prompt
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Say hello!" }
      ],
      max_tokens: 10
    });

    return NextResponse.json({ 
      status: "success",
      message: completion.choices[0]?.message?.content,
      apiKeyConfigured: !!process.env.OPENAI_API_KEY
    });
  } catch (error: any) {
    console.error('OpenAI test error:', error);
    return NextResponse.json(
      { 
        error: "OpenAI API test failed",
        details: error.message,
        apiKeyConfigured: !!process.env.OPENAI_API_KEY
      },
      { status: 500 }
    );
  }
} 