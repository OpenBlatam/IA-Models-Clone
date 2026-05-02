import { NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(request: Request) {
  try {
    const { question, context } = await request.json();

    if (!question) {
      return NextResponse.json(
        { error: "Se requiere una pregunta" },
        { status: 400 }
      );
    }

    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: `Eres un asistente experto en marketing digital de Blatam Academy. 
          Tu objetivo es ayudar a los estudiantes a entender conceptos de marketing de manera clara y concisa. 
          Usa ejemplos prácticos y mantén un tono profesional pero amigable.
          Contexto: ${context || "No hay contexto específico"}`,
        },
        {
          role: "user",
          content: question,
        },
      ],
      temperature: 0.7,
      max_tokens: 500,
    });

    const answer = completion.choices[0]?.message?.content || "Lo siento, no pude generar una respuesta en este momento.";

    return NextResponse.json({ answer });
  } catch (error) {
    console.error("Error processing DeepSeek question:", error);
    return NextResponse.json(
      { error: "Error al procesar la pregunta" },
      { status: 500 }
    );
  }
}  