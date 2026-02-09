import { NextRequest, NextResponse } from 'next/server';

// Configurar timeout máximo para esta ruta (10 minutos)
export const maxDuration = 600; // 10 minutos en segundos
export const dynamic = 'force-dynamic';

const DEEPSEEK_API_KEY = process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY || 'sk-ae1c47feaa3e483b85a936430d1f494a';
const DEEPSEEK_API_BASE_URL = process.env.NEXT_PUBLIC_DEEPSEEK_API_BASE_URL || 'https://api.deepseek.com';
const DEEPSEEK_MODEL = process.env.NEXT_PUBLIC_DEEPSEEK_MODEL || 'deepseek-chat';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { instruction, repository, context } = body;

    console.log('📥 Recibida solicitud de procesamiento:', {
      hasInstruction: !!instruction,
      repository,
      hasContext: !!context,
    });

    if (!instruction) {
      console.error('❌ Error: Se requiere una instrucción');
      return NextResponse.json(
        { error: 'Se requiere una instrucción' },
        { status: 400 }
      );
    }

    // Construir el prompt para DeepSeek - Modo Cursor (ejecución automática)
    const systemPrompt = `Eres un asistente experto en desarrollo de software que funciona como Cursor AI. 
Tu tarea es analizar instrucciones y GENERAR CÓDIGO EJECUTABLE que se implementará automáticamente en repositorios de GitHub.

IMPORTANTE: Debes generar código REAL y COMPLETO que se ejecutará automáticamente. No solo planes, sino implementación real.

Cuando recibas una instrucción:
1. Analiza la instrucción cuidadosamente
2. Genera el código COMPLETO para cada archivo que necesite ser creado o modificado
3. Identifica la ruta exacta de cada archivo
4. Genera código funcional y listo para usar
5. Proporciona el contenido completo de cada archivo

Responde SIEMPRE en formato JSON con la siguiente estructura EXACTA:
{
  "plan": {
    "steps": ["paso 1", "paso 2", ...],
    "files_to_create": [
      {
        "path": "ruta/completa/archivo.ext",
        "content": "contenido completo del archivo",
        "action": "create"
      }
    ],
    "files_to_modify": [
      {
        "path": "ruta/completa/archivo.ext",
        "content": "contenido completo del archivo modificado",
        "action": "update"
      }
    ]
  },
  "commit_message": "Mensaje descriptivo del commit",
  "explanation": "explicación breve de lo que se hizo"
}

CRÍTICO: 
- El campo "content" debe contener el código COMPLETO del archivo, no solo fragmentos
- Usa "files_to_create" para archivos nuevos y "files_to_modify" para archivos existentes
- El commit_message debe ser descriptivo y claro
- Genera código funcional y listo para producción`;

    const userPrompt = `Repositorio: ${repository}
Instrucción: ${instruction}
${context ? `\nContexto adicional: ${JSON.stringify(context, null, 2)}` : ''}

Por favor, analiza esta instrucción y genera un plan de acción detallado.`;

    console.log('📤 Llamando a DeepSeek API...', {
      model: DEEPSEEK_MODEL,
      baseUrl: DEEPSEEK_API_BASE_URL,
      hasApiKey: !!DEEPSEEK_API_KEY,
      apiKeyLength: DEEPSEEK_API_KEY?.length || 0,
    });

    // Llamar a la API de DeepSeek con timeout extendido (10 minutos)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutos

    let response;
    try {
      response = await fetch(`${DEEPSEEK_API_BASE_URL}/v1/chat/completions`, {
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
          temperature: 0.7,
          max_tokens: 4000,
          stream: false,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
    } catch (fetchError: any) {
      clearTimeout(timeoutId);
      if (fetchError.name === 'AbortError') {
        console.error('Timeout al llamar a DeepSeek API');
        return NextResponse.json(
          { error: 'La solicitud tardó demasiado tiempo. Por favor, intenta con una instrucción más corta.' },
          { status: 408 }
        );
      }
      throw fetchError;
    }

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error de DeepSeek API:', {
        status: response.status,
        statusText: response.statusText,
        error: errorText,
      });
      
      let errorMessage = 'Error al comunicarse con DeepSeek API';
      try {
        const errorData = JSON.parse(errorText);
        errorMessage = errorData.error?.message || errorData.message || errorMessage;
      } catch {
        errorMessage = errorText || errorMessage;
      }
      
      return NextResponse.json(
        { error: errorMessage },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log('📥 Respuesta de DeepSeek recibida:', {
      hasChoices: !!data.choices,
      choicesLength: data.choices?.length,
      hasContent: !!data.choices?.[0]?.message?.content,
      contentLength: data.choices?.[0]?.message?.content?.length || 0,
    });
    
    const content = data.choices?.[0]?.message?.content || '';
    
    if (!content) {
      console.warn('⚠️ No se recibió contenido de DeepSeek');
      console.warn('Respuesta completa:', JSON.stringify(data, null, 2));
      return NextResponse.json(
        { error: 'No se recibió respuesta de DeepSeek. Por favor, intenta de nuevo.' },
        { status: 500 }
      );
    }
    
    console.log('✅ Contenido recibido de DeepSeek:', content.substring(0, 200) + '...');

    // Intentar parsear JSON de la respuesta
    let plan = null;
    let code = null;
    let explanation = content;

    try {
      // Buscar JSON en la respuesta
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        plan = parsed.plan || null;
        code = parsed.code || null;
        explanation = parsed.explanation || content;
      }
    } catch (e) {
      // Si no se puede parsear, usar el contenido completo
      console.log('No se pudo parsear JSON, usando contenido completo');
    }

    const responseData = {
      success: true,
      content: explanation,
      plan: plan || {
        steps: [explanation],
        files_to_create: [],
        files_to_modify: [],
      },
      code: code || null,
      usage: data.usage,
    };
    
    console.log('✅ Procesamiento completado exitosamente');
    return NextResponse.json(responseData);
  } catch (error: any) {
    console.error('❌ Error processing instruction:', error);
    console.error('Error details:', {
      message: error.message,
      stack: error.stack,
      name: error.name,
    });
    
    const errorMessage = error.message || 'Error al procesar la instrucción';
    return NextResponse.json(
      { 
        success: false,
        error: errorMessage,
        content: '',
      },
      { status: 500 }
    );
  }
}

