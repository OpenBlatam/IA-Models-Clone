import { NextRequest } from 'next/server';

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

    console.log('📥 Recibida solicitud de streaming:', {
      hasInstruction: !!instruction,
      repository,
      hasContext: !!context,
    });

    if (!instruction) {
      return new Response(
        JSON.stringify({ error: 'Se requiere una instrucción' }),
        { 
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        }
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

    console.log('📤 Llamando a DeepSeek API con streaming...', {
      model: DEEPSEEK_MODEL,
      baseUrl: DEEPSEEK_API_BASE_URL,
      hasApiKey: !!DEEPSEEK_API_KEY,
    });

    // Crear un ReadableStream para el streaming
    console.log('🔄 [DEEPSEEK] Creando ReadableStream...');
    const stream = new ReadableStream({
      async start(controller) {
        console.log('🚀 [DEEPSEEK] Método start() llamado, iniciando stream...');
        try {
          // Llamar a la API de DeepSeek con streaming habilitado
          // Sin timeout - dejar que el stream continúe indefinidamente
          const abortController = new AbortController();
          
          console.log('📤 [DEEPSEEK] Llamada a DeepSeek API:', {
            model: DEEPSEEK_MODEL,
            instructionLength: userPrompt.length,
            systemPromptLength: systemPrompt.length,
            maxTokens: 8000,
            baseUrl: DEEPSEEK_API_BASE_URL,
            apiKey: DEEPSEEK_API_KEY ? `${DEEPSEEK_API_KEY.substring(0, 10)}...` : 'NO KEY',
          });
          
          console.log('📤 [DEEPSEEK] Realizando fetch a DeepSeek...');
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
              temperature: 0.7,
              max_tokens: 8000, // Aumentado de 4000 a 8000 para respuestas más largas
              stream: true, // Habilitar streaming
            }),
            signal: abortController.signal,
            // No agregar timeout - el stream debe continuar hasta que termine naturalmente
          });

          console.log('📥 [DEEPSEEK] Respuesta recibida:', {
            status: response.status,
            statusText: response.statusText,
            ok: response.ok,
            hasBody: !!response.body,
            headers: Object.fromEntries(response.headers.entries()),
          });

          if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = 'Error al comunicarse con DeepSeek API';
            try {
              const errorData = JSON.parse(errorText);
              errorMessage = errorData.error?.message || errorData.message || errorMessage;
            } catch {
              errorMessage = errorText || errorMessage;
            }
            
            console.error('❌ [DEEPSEEK] Error en respuesta de DeepSeek API:', {
              status: response.status,
              statusText: response.statusText,
              error: errorMessage,
              errorText: errorText.substring(0, 500),
            });
            
            controller.enqueue(
              new TextEncoder().encode(
                JSON.stringify({ error: errorMessage, done: true }) + '\n'
              )
            );
            controller.close();
            return;
          }

          if (!response.body) {
            console.error('❌ [DEEPSEEK] La respuesta de DeepSeek no tiene body');
            console.error('❌ [DEEPSEEK] Detalles de la respuesta:', {
              status: response.status,
              statusText: response.statusText,
              headers: Object.fromEntries(response.headers.entries()),
            });
            controller.enqueue(
              new TextEncoder().encode(
                JSON.stringify({ error: 'No se recibió respuesta de DeepSeek', done: true }) + '\n'
              )
            );
            controller.close();
            return;
          }
          
          console.log('✅ [DEEPSEEK] Stream de DeepSeek iniciado correctamente, comenzando lectura...');
          console.log('✅ [DEEPSEEK] Content-Type:', response.headers.get('content-type'));
          console.log('✅ [DEEPSEEK] Transfer-Encoding:', response.headers.get('transfer-encoding'));

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          let lastActivity = Date.now();
          let heartbeatInterval: NodeJS.Timeout | null = null;

          // Heartbeat para mantener la conexión viva - enviar cada 30 segundos
          heartbeatInterval = setInterval(() => {
            try {
              const now = Date.now();
              controller.enqueue(
                new TextEncoder().encode(
                  JSON.stringify({ type: 'heartbeat', timestamp: now }) + '\n'
                )
              );
              lastActivity = now; // Actualizar última actividad
            } catch (e) {
              // Si el controller está cerrado, limpiar el intervalo
              if (heartbeatInterval) {
                clearInterval(heartbeatInterval);
                heartbeatInterval = null;
              }
            }
          }, 30000); // Enviar heartbeat cada 30 segundos para mantener la conexión viva

          let totalContentLength = 0;
          let receivedDoneSignal = false;
          
          try {
            let deepseekChunkCount = 0;
            let emptyChunkCount = 0;
            const maxEmptyChunks = 10; // Permitir hasta 10 chunks vacíos consecutivos antes de considerar error
            
            while (true) {
              const { done, value } = await reader.read();
              
              if (done) {
                console.log(`✅ Stream de DeepSeek terminó. Contenido total enviado: ${totalContentLength} caracteres, ${deepseekChunkCount} chunks recibidos`);
                clearInterval(heartbeatInterval);
                
                // Procesar cualquier buffer restante
                if (buffer.trim()) {
                  console.log(`📥 Procesando buffer restante (${buffer.length} caracteres)`);
                  const remainingLines = buffer.split('\n').filter(line => line.trim());
                  for (const line of remainingLines) {
                    if (line.startsWith('data: ')) {
                      const data = line.slice(6);
                      if (data !== '[DONE]') {
                        try {
                          const json = JSON.parse(data);
                          const content = json.choices?.[0]?.delta?.content || '';
                          if (content) {
                            totalContentLength += content.length;
                            const jsonToSend = JSON.stringify({ content, done: false }) + '\n';
                            controller.enqueue(new TextEncoder().encode(jsonToSend));
                          }
                        } catch (e) {
                          console.warn('Error procesando buffer restante:', e);
                        }
                      }
                    }
                  }
                }
                
                // Si no se envió contenido, podría ser un error
                if (totalContentLength === 0) {
                  console.error(`❌ Stream de DeepSeek terminó sin enviar contenido`);
                  console.error(`❌ Esto podría indicar un error en la API de DeepSeek o en el procesamiento del stream`);
                  console.error(`❌ Chunks recibidos: ${deepseekChunkCount}, Buffer final: ${buffer.length} caracteres`);
                  console.error(`❌ Buffer contenido: "${buffer.substring(0, 200)}"`);
                  
                  // Verificar si el buffer contiene algún error de DeepSeek
                  if (buffer.includes('error') || buffer.includes('Error')) {
                    try {
                      const errorMatch = buffer.match(/"error":\s*\{[^}]*"message":\s*"([^"]+)"/);
                      if (errorMatch) {
                        const errorMessage = errorMatch[1];
                        console.error(`❌ Error de DeepSeek encontrado: ${errorMessage}`);
                        controller.enqueue(
                          new TextEncoder().encode(
                            JSON.stringify({ 
                              error: `Error de DeepSeek: ${errorMessage}`,
                              done: true 
                            }) + '\n'
                          )
                        );
                        controller.close();
                        return;
                      }
                    } catch (e) {
                      // Continuar con el error genérico
                    }
                  }
                  
                  // Enviar un error al cliente
                  controller.enqueue(
                    new TextEncoder().encode(
                      JSON.stringify({ 
                        error: 'Stream terminó sin contenido. Esto podría indicar un problema con la API de DeepSeek. Verifica la API key y los parámetros de la solicitud.',
                        done: true 
                      }) + '\n'
                    )
                  );
                  controller.close();
                  return;
                }
                
                // Esperar un momento para asegurar que todo el contenido se haya procesado
                await new Promise(resolve => setTimeout(resolve, 500));
                
                // Enviar señal de finalización solo después de procesar todo
                controller.enqueue(
                  new TextEncoder().encode(
                    JSON.stringify({ done: true, totalContent: totalContentLength }) + '\n'
                  )
                );
                controller.close();
                break;
              }
              
              deepseekChunkCount++;
              
              // Verificar que value no sea null/undefined
              if (!value || value.length === 0) {
                emptyChunkCount++;
                if (deepseekChunkCount <= 5) {
                  console.warn(`⚠️ Chunk #${deepseekChunkCount} recibido con value vacío (${emptyChunkCount} vacíos consecutivos)`);
                }
                
                // Si recibimos demasiados chunks vacíos consecutivos, podría ser un problema
                if (emptyChunkCount >= maxEmptyChunks && totalContentLength === 0) {
                  console.error(`❌ Demasiados chunks vacíos consecutivos (${emptyChunkCount}), posible problema con el stream`);
                  console.error(`❌ Buffer actual: "${buffer.substring(0, 200)}"`);
                  // Continuar intentando, pero registrar el problema
                }
                
                // Resetear contador si recibimos contenido
                if (totalContentLength > 0) {
                  emptyChunkCount = 0;
                }
                
                continue;
              }
              
              // Resetear contador de chunks vacíos cuando recibimos contenido
              if (value.length > 0) {
                emptyChunkCount = 0;
              }
              
              if (deepseekChunkCount <= 5 || deepseekChunkCount % 50 === 0) {
                console.log(`📥 Stream de DeepSeek - Chunk #${deepseekChunkCount} recibido (${value.length} bytes)`);
              }

              lastActivity = Date.now();
              
              // Decodificar el chunk
              const decodedChunk = decoder.decode(value, { stream: true });
              if (deepseekChunkCount <= 5) {
                console.log(`📥 Chunk decodificado: ${decodedChunk.length} caracteres, preview: ${decodedChunk.substring(0, 100)}`);
              }
              
              // Agregar al buffer
              buffer += decodedChunk;
              
              // Procesar líneas completas
              const lines = buffer.split('\n');
              // Mantener la última línea incompleta en el buffer
              buffer = lines.pop() || '';
              
              if (deepseekChunkCount <= 5) {
                console.log(`📥 Buffer después de procesar: ${buffer.length} caracteres, ${lines.length} líneas completas`);
              }

              for (const line of lines) {
                if (line.trim() === '') continue;
                if (line.startsWith('data: ')) {
                  const data = line.slice(6);
                  if (data === '[DONE]') {
                    // Marcar que recibimos [DONE] pero NO cerrar todavía
                    console.log('📤 [DONE] recibido de DeepSeek, pero continuando hasta que el stream termine naturalmente');
                    receivedDoneSignal = true;
                    // Continuar procesando - no cerrar todavía
                    continue;
                  }

                  try {
                    const json = JSON.parse(data);
                    const content = json.choices?.[0]?.delta?.content || '';
                    
                    if (content) {
                      totalContentLength += content.length;
                      if (deepseekChunkCount <= 5 || deepseekChunkCount % 50 === 0) {
                        console.log('📤 Enviando chunk al cliente:', {
                          chunkNumber: deepseekChunkCount,
                          contentLength: content.length,
                          totalContentLength: totalContentLength,
                          contentPreview: content.substring(0, 50),
                        });
                      }
                      const jsonToSend = JSON.stringify({ content, done: false }) + '\n';
                      controller.enqueue(
                        new TextEncoder().encode(jsonToSend)
                      );
                    } else {
                      // Verificar si hay finish_reason u otra señal de finalización
                      if (json.choices?.[0]?.finish_reason) {
                        console.log('📤 Señal de finalización recibida:', json.choices[0].finish_reason);
                        console.log(`📤 Contenido acumulado hasta ahora: ${totalContentLength} caracteres`);
                        console.log('📤 Continuando procesamiento - NO cerrar stream todavía');
                        receivedDoneSignal = true;
                        // NO enviar done todavía, esperar a que el stream termine naturalmente
                        // Continuar procesando para asegurar que se reciba todo el contenido
                      } else {
                        // Si no hay contenido ni finish_reason, podría ser un chunk vacío
                        // Log para debugging solo en los primeros chunks
                        if (deepseekChunkCount <= 10) {
                          console.log('📤 Chunk sin contenido ni finish_reason:', {
                            chunkNumber: deepseekChunkCount,
                            hasChoices: !!json.choices,
                            choicesLength: json.choices?.length,
                            delta: json.choices?.[0]?.delta,
                            fullJson: JSON.stringify(json).substring(0, 200),
                          });
                        }
                      }
                    }
                  } catch (e) {
                    // Ignorar errores de parsing de líneas individuales
                    console.warn('⚠️ Error parsing línea del stream:', e);
                  }
                }
              }
              
              // Si recibimos [DONE] pero el stream aún no terminó, continuar leyendo
              if (receivedDoneSignal && !done) {
                console.log('📤 Stream aún activo después de [DONE], continuando lectura...');
              }
            }
          } catch (streamError: any) {
            if (heartbeatInterval) {
              clearInterval(heartbeatInterval);
              heartbeatInterval = null;
            }
            console.error('Error en el stream:', streamError);
            controller.enqueue(
              new TextEncoder().encode(
                JSON.stringify({ error: streamError.message || 'Error en el stream', done: true }) + '\n'
              )
            );
            controller.close();
          }
        } catch (error: any) {
          console.error('❌ Error processing stream:', error);
          try {
            controller.enqueue(
              new TextEncoder().encode(
                JSON.stringify({ 
                  error: error.message || 'Error al procesar el stream',
                  done: true 
                }) + '\n'
              )
            );
            controller.close();
          } catch (closeError) {
            // El controller ya está cerrado, no hacer nada
            console.log('Controller ya estaba cerrado');
          }
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no', // Deshabilitar buffering de nginx
        'Keep-Alive': 'timeout=600', // 10 minutos
      },
    });
  } catch (error: any) {
    console.error('❌ Error creating stream:', error);
    return new Response(
      JSON.stringify({ 
        success: false,
        error: error.message || 'Error al crear el stream',
        done: true
      }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}



