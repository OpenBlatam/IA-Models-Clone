import { NextRequest, NextResponse } from 'next/server';

const DEEPSEEK_API_KEY = process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY || 'sk-ae1c47feaa3e483b85a936430d1f494a';
const DEEPSEEK_API_BASE_URL = process.env.NEXT_PUBLIC_DEEPSEEK_API_BASE_URL || 'https://api.deepseek.com';
const DEEPSEEK_MODEL = process.env.NEXT_PUBLIC_DEEPSEEK_MODEL || 'deepseek-chat';

export async function POST(request: NextRequest) {
  try {
    const { description, language = 'typescript', context } = await request.json();

    if (!description) {
      return NextResponse.json(
        { error: 'Se requiere una descripción' },
        { status: 400 }
      );
    }

    const systemPrompt = `Eres un experto programador. Genera código limpio, bien estructurado y bien documentado.
Responde SOLO con el código solicitado, sin explicaciones adicionales a menos que se solicite específicamente.`;

    const userPrompt = `Genera código en ${language} que:
${description}

${context ? `\nContexto adicional:\n${context}` : ''}

Responde SOLO con el código, sin explicaciones.`;

    const response = await fetch(`${DEEPSEEK_API_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        messages: [
          {
            role: 'system',
            content: systemPrompt,
          },
          {
            role: 'user',
            content: userPrompt,
          },
        ],
        temperature: 0.3,
        max_tokens: 4000,
        stream: false,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error de DeepSeek API:', errorText);
      return NextResponse.json(
        { error: 'Error al comunicarse con DeepSeek API' },
        { status: response.status }
      );
    }

    const data = await response.json();
    const code = data.choices?.[0]?.message?.content || '';

    return NextResponse.json({
      success: true,
      code: code.trim(),
      usage: data.usage,
    });
  } catch (error: any) {
    console.error('Error generating code:', error);
    return NextResponse.json(
      { error: error.message || 'Error al generar código' },
      { status: 500 }
    );
  }
}



