import { NextRequest, NextResponse } from 'next/server';

const OPENROUTER_API_KEY = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY || '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';
const OPENROUTER_API_BASE_URL = 'https://openrouter.ai/api/v1';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { instruction, repository, model, context } = body;

    if (!instruction) {
      return NextResponse.json({ error: 'Se requiere una instrucción' }, { status: 400 });
    }

    if (!model) {
      return NextResponse.json({ error: 'Se requiere un modelo' }, { status: 400 });
    }

    if (!OPENROUTER_API_KEY) {
      return NextResponse.json({ error: 'OpenRouter API Key no configurada' }, { status: 500 });
    }

    const systemPrompt = `Eres un asistente experto en desarrollo de software y automatización de GitHub. 
Tu tarea es analizar instrucciones y generar planes de acción detallados para implementar cambios en repositorios de GitHub.

Cuando recibas una instrucción:
1. Analiza la instrucción cuidadosamente
2. Genera un plan de acción paso a paso
3. Identifica qué archivos necesitan ser creados o modificados
4. Genera el código necesario cuando sea apropiado
5. Proporciona instrucciones claras y ejecutables

Responde en formato JSON con la siguiente estructura:
{
  "plan": {
    "steps": ["paso 1", "paso 2", ...],
    "files_to_create": ["archivo1", "archivo2", ...],
    "files_to_modify": ["archivo1", "archivo2", ...]
  },
  "code": "código generado si aplica",
  "explanation": "explicación del plan"
}`;

    const userPrompt = `Repositorio: ${repository}
Instrucción: ${instruction}
${context ? `\nContexto adicional: ${JSON.stringify(context, null, 2)}` : ''}

Por favor, genera el plan, el código y la explicación en el formato JSON especificado.`;

    const openrouterResponse = await fetch(`${OPENROUTER_API_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'HTTP-Referer': process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
        'X-Title': 'bulk',
      },
      body: JSON.stringify({
        model: model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.7,
        max_tokens: 4000,
        stream: true,
      }),
      signal: request.signal,
    });

    if (!openrouterResponse.ok) {
      const errorText = await openrouterResponse.text();
      console.error('Error de OpenRouter API:', openrouterResponse.status, errorText);
      return NextResponse.json(
        { error: `Error al comunicarse con OpenRouter API: ${openrouterResponse.status} - ${errorText}` },
        { status: openrouterResponse.status }
      );
    }

    const stream = new ReadableStream({
      async start(controller) {
        const reader = openrouterResponse.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          controller.error('No se pudo obtener el lector del cuerpo de la respuesta.');
          return;
        }

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              controller.enqueue(JSON.stringify({ done: true }) + '\n');
              break;
            }
            const chunk = decoder.decode(value, { stream: true });
            // OpenRouter envía chunks como "data: { ... }\n\n"
            chunk.split('\n').forEach(line => {
              if (line.startsWith('data: ')) {
                const data = line.substring(6);
                if (data === '[DONE]') {
                  controller.enqueue(JSON.stringify({ done: true }) + '\n');
                  return;
                }
                try {
                  const parsed = JSON.parse(data);
                  const content = parsed.choices?.[0]?.delta?.content || '';
                  if (content) {
                    controller.enqueue(JSON.stringify({ content }) + '\n');
                  }
                } catch (e) {
                  console.error('Error parsing OpenRouter stream chunk:', e, data);
                }
              }
            });
          }
        } catch (error: any) {
          if (error.name === 'AbortError') {
            console.log('Streaming abortado por el cliente.');
          } else {
            console.error('Error leyendo el stream de OpenRouter:', error);
          }
          controller.error(error);
        } finally {
          reader.releaseLock();
          controller.close();
        }
      },
    });

    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
      },
    });

  } catch (error: any) {
    console.error('Error en la ruta /api/openrouter/stream:', error);
    return NextResponse.json(
      { error: error.message || 'Error interno del servidor' },
      { status: 500 }
    );
  }
}



