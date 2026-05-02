import { NextRequest, NextResponse } from "next/server";
import axios from "axios";
import * as cheerio from "cheerio";
import { extractColors, extractFonts } from "@/lib/utils";

// Remove edge runtime
// export const runtime = "edge";

// Validate DeepSeek configuration
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;

const DUMMY_BRAND_KIT = `1. PALETA DE COLORES:
- #FAF3F3, #FF6FA5, #FFD6E0, #B2F0F0, #F39AC1

2. TIPOGRAFÍA:
- Montserrat, Roboto, Inter

3. TONO DE VOZ:
- Innovador, Cercano, Inspirador

4. VALORES DE MARCA:
- Creatividad, Tecnología, Educación

5. ELEMENTOS VISUALES:
- Imágenes limpias, iconos modernos, estilo minimalista`;

const BLOG_TEMPLATES = {
  guide: {
    sections: [
      {
        title: "Introducción",
        type: "intro",
        content: "Introduce el tema y su importancia en el contexto actual. Explica por qué es relevante para la audiencia objetivo."
      },
      {
        title: "¿Qué es {topic}?",
        type: "definition",
        content: "Define el concepto de manera clara y concisa. Incluye ejemplos prácticos y casos de uso."
      },
      {
        title: "Beneficios de {topic}",
        type: "benefits",
        content: "Lista los principales beneficios y ventajas. Apoya cada punto con datos o ejemplos concretos."
      },
      {
        title: "Cómo implementar {topic}",
        type: "how-to",
        content: "Proporciona pasos detallados y accionables. Incluye consejos prácticos y mejores prácticas."
      },
      {
        title: "Mejores prácticas",
        type: "best-practices",
        content: "Comparte recomendaciones basadas en experiencia. Evita errores comunes y optimiza resultados."
      },
      {
        title: "Conclusión",
        type: "conclusion",
        content: "Resume los puntos clave y proporciona un llamado a la acción claro."
      }
    ],
    metadata: {
      keywords: ["guía", "tutorial", "aprender", "implementar"],
      readingTime: "15-20 min",
      category: "Tutoriales"
    }
  },
  analysis: {
    sections: [
      {
        title: "Contexto",
        type: "context",
        content: "Establece el marco de referencia y la situación actual del tema a analizar."
      },
      {
        title: "Análisis del mercado",
        type: "market-analysis",
        content: "Examina las tendencias, competidores y oportunidades en el mercado."
      },
      {
        title: "Tendencias actuales",
        type: "trends",
        content: "Identifica y analiza las tendencias más relevantes del momento."
      },
      {
        title: "Casos de estudio",
        type: "case-studies",
        content: "Presenta ejemplos reales y sus resultados. Extrae lecciones aprendidas."
      },
      {
        title: "Recomendaciones",
        type: "recommendations",
        content: "Ofrece sugerencias prácticas basadas en el análisis realizado."
      },
      {
        title: "Perspectivas futuras",
        type: "future",
        content: "Proyecta escenarios futuros y oportunidades de crecimiento."
      }
    ],
    metadata: {
      keywords: ["análisis", "mercado", "tendencias", "estudio"],
      readingTime: "20-25 min",
      category: "Análisis"
    }
  },
  tips: {
    sections: [
      {
        title: "Introducción",
        type: "intro",
        content: "Presenta el tema y la importancia de los tips que se compartirán."
      },
      {
        title: "Tip #1: {tip1}",
        type: "tip",
        content: "Explica el primer tip con ejemplos prácticos y casos de uso."
      },
      {
        title: "Tip #2: {tip2}",
        type: "tip",
        content: "Detalla el segundo tip con pasos específicos y resultados esperados."
      },
      {
        title: "Tip #3: {tip3}",
        type: "tip",
        content: "Describe el tercer tip con consejos adicionales y mejores prácticas."
      },
      {
        title: "Consejos adicionales",
        type: "additional",
        content: "Comparte recomendaciones extra y trucos avanzados."
      },
      {
        title: "Resumen",
        type: "summary",
        content: "Recapitula los tips principales y su aplicación práctica."
      }
    ],
    metadata: {
      keywords: ["tips", "trucos", "consejos", "mejores prácticas"],
      readingTime: "10-15 min",
      category: "Consejos"
    }
  },
  trends: {
    sections: [
      {
        title: "Panorama actual",
        type: "overview",
        content: "Describe el estado actual del sector y las fuerzas que lo moldean."
      },
      {
        title: "Tendencia #1: {trend1}",
        type: "trend",
        content: "Analiza la primera tendencia, su impacto y casos de éxito."
      },
      {
        title: "Tendencia #2: {trend2}",
        type: "trend",
        content: "Examina la segunda tendencia, oportunidades y desafíos."
      },
      {
        title: "Tendencia #3: {trend3}",
        type: "trend",
        content: "Explora la tercera tendencia, innovaciones y aplicaciones prácticas."
      },
      {
        title: "Impacto en la industria",
        type: "impact",
        content: "Evalúa el impacto de las tendencias en diferentes sectores."
      },
      {
        title: "Conclusión",
        type: "conclusion",
        content: "Sintetiza las tendencias clave y su relevancia futura."
      }
    ],
    metadata: {
      keywords: ["tendencias", "innovación", "futuro", "tecnología"],
      readingTime: "12-18 min",
      category: "Tendencias"
    }
  }
};

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { type, template, topic, audience, brandKit, structure } = body;

    if (!DEEPSEEK_API_KEY) {
      return NextResponse.json(
        { error: "DeepSeek API key not configured" },
        { status: 500 }
      );
    }

    if (type === "blog") {
      const selectedTemplate = BLOG_TEMPLATES[template as keyof typeof BLOG_TEMPLATES];
      if (!selectedTemplate) {
        return NextResponse.json(
          { error: "Template not found" },
          { status: 400 }
        );
      }

      // Personalizar el template con el tema y la audiencia
      const customizedSections = selectedTemplate.sections.map(section => ({
        ...section,
        title: section.title.replace(/{topic}/g, topic),
        content: section.content.replace(/{topic}/g, topic)
      }));

      // Generar el contenido usando DeepSeek
      const promptText = `Genera un post de blog optimizado para SEO con el siguiente formato:
      - Título atractivo y optimizado para SEO (máximo 60 caracteres)
      - Extracto conciso (máximo 160 caracteres)
      - Contenido estructurado en secciones: ${customizedSections.map(s => s.title).join(", ")}
      - Palabras clave relevantes (máximo 5)
      - Tiempo de lectura estimado
      - Meta descripción para SEO
      
      Tema: ${topic}
      Audiencia objetivo: ${audience}
      Brand Kit: ${JSON.stringify(brandKit)}
      
      El post debe ser:
      - Bien estructurado y fácil de leer
      - Optimizado para SEO
      - Con ejemplos prácticos
      - Con datos y estadísticas relevantes
      - Adaptado al tono de la marca`;

      const response = await fetch("https://api.deepseek.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${DEEPSEEK_API_KEY}`
        },
        body: JSON.stringify({
          model: "deepseek-chat",
          messages: [
            {
              role: "system",
              content: "Eres un experto en marketing digital y creación de contenido optimizado para SEO."
            },
            {
              role: "user",
              content: promptText
            }
          ],
          temperature: 0.7,
          max_tokens: 2000
        })
      });

      if (!response.ok) {
        throw new Error("Error al generar el contenido");
      }

      const data = await response.json();
      const generatedContent = data.choices[0].message.content;

      // Procesar y estructurar el contenido generado
      const sections = customizedSections.map(section => ({
        title: section.title,
        content: generatedContent
      }));

      return NextResponse.json({
        title: `Guía completa sobre ${topic}`,
        excerpt: `Aprende todo lo que necesitas saber sobre ${topic} en esta guía detallada.`,
        content: {
          sections
        },
        metadata: {
          ...selectedTemplate.metadata,
          keywords: [...selectedTemplate.metadata.keywords, topic.toLowerCase()],
          seoDescription: `Descubre todo sobre ${topic} en esta guía completa. Aprende las mejores prácticas y consejos expertos.`
        }
      });
    }

    if (type === "facebook-post") {
      // Generate Facebook post content based on template
      const templates = {
        "Producto Nuevo": {
          title: "¡Novedad Exclusiva! 🎉",
          content: `¡Estamos emocionados de presentar nuestro último lanzamiento! ✨

Este producto revolucionario cambiará la forma en que [beneficio principal]. Diseñado con pasión y atención al detalle, cada característica ha sido pensada para mejorar tu experiencia.`,
          imageUrl: "https://images.unsplash.com/photo-1493612276216-ee3925520721",
          callToAction: "¡Sé de los primeros en probarlo!",
          hashtags: ["#NuevoLanzamiento", "#Innovación", "#Exclusivo", "#Tendencias", "#Calidad"]
        },
        "Oferta Especial": {
          title: "🔥 ¡Oferta Relámpago! ⚡",
          content: `¡No dejes pasar esta oportunidad única! 💰

Por tiempo limitado, disfruta de un [descuento] en todos nuestros productos. Esta oferta especial es tu chance de obtener lo mejor a un precio increíble.`,
          imageUrl: "https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da",
          callToAction: "¡Aprovecha ahora!",
          hashtags: ["#OfertaEspecial", "#Descuento", "#Oportunidad", "#Ahorra", "#LimitedTime"]
        },
        "Contenido Educativo": {
          title: "📚 Tips Pro: [Tema]",
          content: `¿Sabías que [dato interesante]? 💡

En este post te compartimos los mejores consejos para [beneficio]. Aplica estos tips y verás resultados inmediatos.`,
          imageUrl: "https://images.unsplash.com/photo-1501504905252-473c47e087f8",
          callToAction: "¡Comparte si te ayudó!",
          hashtags: ["#TipsPro", "#Aprende", "#Consejos", "#Educación", "#Desarrollo"]
        },
        "Historia de Éxito": {
          title: "🌟 Historia Inspiradora: [Nombre]",
          content: `Conoce la increíble historia de [nombre] y cómo [producto/servicio] transformó su vida. 💪

De [situación inicial] a [resultado final], este es un testimonio real de superación y éxito.`,
          imageUrl: "https://images.unsplash.com/photo-1551836022-d5d88e9218df",
          callToAction: "¡Cuéntanos tu historia!",
          hashtags: ["#HistoriaDeÉxito", "#Inspiración", "#Testimonio", "#Superación", "#Éxito"]
        }
      };

      const template = templates[template as keyof typeof templates] || templates["Producto Nuevo"];
      
      // Customize template with brand kit data if available
      if (brandKit) {
        template.title = template.title.replace("[Tema]", brandKit.title || "Nuestro Producto");
        template.content = template.content
          .replace("[beneficio principal]", brandKit.description || "usar nuestro producto")
          .replace("[descuento]", "20%")
          .replace("[dato interesante]", "nuestros clientes aumentan su productividad en un 50%")
          .replace("[beneficio]", "optimizar tu trabajo")
          .replace("[nombre]", brandKit.title || "Nuestro Cliente")
          .replace("[producto/servicio]", brandKit.title || "nuestro servicio")
          .replace("[situación inicial]", "comenzar desde cero")
          .replace("[resultado final]", "alcanzar el éxito");
      }

      return NextResponse.json({ facebookPost: template });
    }

    if (type === "blog-post") {
      const selectedTemplate = BLOG_TEMPLATES[template as keyof typeof BLOG_TEMPLATES];
      if (!selectedTemplate) {
        return NextResponse.json(
          { error: "Template not found" },
          { status: 400 }
        );
      }

      // Personalizar el template con el tema y la audiencia
      const customizedSections = selectedTemplate.sections.map(section => ({
        ...section,
        title: section.title.replace(/{topic}/g, topic),
        content: section.content.replace(/{topic}/g, topic)
      }));

      // Generar el contenido usando DeepSeek
      const promptText = `Genera un post de blog optimizado para SEO con el siguiente formato:
      - Título atractivo y optimizado para SEO (máximo 60 caracteres)
      - Extracto conciso (máximo 160 caracteres)
      - Contenido estructurado en secciones: ${customizedSections.map(s => s.title).join(", ")}
      - Palabras clave relevantes (máximo 5)
      - Tiempo de lectura estimado
      - Meta descripción para SEO
      
      Tema: ${topic}
      Audiencia objetivo: ${audience}
      Brand Kit: ${JSON.stringify(brandKit)}
      
      El post debe ser:
      - Bien estructurado y fácil de leer
      - Optimizado para SEO
      - Con ejemplos prácticos
      - Con datos y estadísticas relevantes
      - Adaptado al tono de la marca`;

      const response = await fetch("https://api.deepseek.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${DEEPSEEK_API_KEY}`
        },
        body: JSON.stringify({
          model: "deepseek-chat",
          messages: [
            {
              role: "system",
              content: "Eres un experto en marketing digital y creación de contenido optimizado para SEO."
            },
            {
              role: "user",
              content: promptText
            }
          ],
          temperature: 0.7,
          max_tokens: 2000
        })
      });

      if (!response.ok) {
        throw new Error("Error al generar el contenido");
      }

      const data = await response.json();
      const generatedContent = data.choices[0].message.content;

      // Procesar y estructurar el contenido generado
      const sections = customizedSections.map(section => ({
        title: section.title,
        content: generatedContent
      }));

      return NextResponse.json({
        title: `Guía completa sobre ${topic}`,
        excerpt: `Aprende todo lo que necesitas saber sobre ${topic} en esta guía detallada.`,
        content: {
          sections
        },
        metadata: {
          ...selectedTemplate.metadata,
          keywords: [...selectedTemplate.metadata.keywords, topic.toLowerCase()],
          seoDescription: `Descubre todo sobre ${topic} en esta guía completa. Aprende las mejores prácticas y consejos expertos.`
        }
      });
    }

    // Check DeepSeek configuration
    if (!DEEPSEEK_API_KEY) {
      return NextResponse.json(
        { 
          error: "Error de configuración",
          details: "La API key de DeepSeek no está configurada"
        },
        { 
          status: 500,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
          }
        }
      );
    }

    if (!topic) {
      return NextResponse.json(
        { error: "Topic required" }, 
        { 
          status: 400,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
          }
        }
      );
    }

    // Validate URL format
    try {
      new URL(topic);
    } catch (e) {
      return NextResponse.json(
        { error: "Invalid URL" }, 
        { 
          status: 400,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
          }
        }
      );
    }

    // 1. Scraping con axios y cheerio
    let extracted = "";
    let brandElements = {
      colors: [] as string[],
      fonts: [] as string[],
      logos: [] as string[],
      taglines: [] as string[],
    };
    let scrapingOk = false;

    try {
      const { data } = await axios.get(topic, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        },
        timeout: 10000 // 10 second timeout
      });
      const $ = cheerio.load(data);
      
      // Extract brand elements
      $('style, [style]').each((_, el) => {
        const style = $(el).attr('style') || $(el).text();
        // Extract colors
        const colorMatches = style.match(/#[0-9a-fA-F]{3,6}|rgb\([^)]+\)|rgba\([^)]+\)/g) || [];
        brandElements.colors.push(...colorMatches);
        
        // Extract fonts
        const fontMatches = style.match(/font-family:\s*([^;]+)/g) || [];
        brandElements.fonts.push(...fontMatches.map(f => f.replace('font-family:', '').trim()));
      });

      // Extract logos
      $('img[src*="logo"], img[alt*="logo"]').each((_, el) => {
        const src = $(el).attr('src');
        if (src) brandElements.logos.push(src);
      });

      // Extract taglines
      $('h1, h2, .tagline, .slogan').each((_, el) => {
        const text = $(el).text().trim();
        if (text) brandElements.taglines.push(text);
      });

      const title = $('title').text().trim();
      const description = $('meta[name="description"]').attr('content')?.trim() || '';
      const h1 = $('h1').first().text().trim();
      const h2 = $('h2').first().text().trim();
      const mainContent = $('main, article, .content, #content').first().text().trim().slice(0, 500);
      
      extracted = [title, description, h1, h2, mainContent]
        .filter(Boolean)
        .join(' | ');

      scrapingOk = !!extracted && extracted.length > 10;

    } catch (scrapeError) {
      console.error('Error scraping URL:', scrapeError);
      // No return, fallback to generic prompt
    }

    // Call our DeepSeek backend
    try {
      const apiUrl = (process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8001').replace(/\/$/, '');
      const fullUrl = `${apiUrl}/api/ads-ia`;
      console.log('Calling backend at:', fullUrl);
      
      const response = await axios.post(
        fullUrl,
        {
          url: topic,
          type,
          prompt: extracted,
          website_content: extracted,
          brand_elements: brandElements
        },
        {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 30000, // 30 second timeout
          family: 4 // Force IPv4
        }
      );

      return NextResponse.json(
        response.data,
        {
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
          }
        }
      );
    } catch (apiError: any) {
      console.error('Error calling DeepSeek API:', apiError);
      console.error('Error details:', {
        message: apiError.message,
        code: apiError.code,
        address: apiError.address,
        port: apiError.port,
        config: apiError.config
      });
      
      if (type === "brand-kit") {
        return NextResponse.json(
          { brandKit: DUMMY_BRAND_KIT, fallback: true, error: "API error", details: apiError.message },
          { 
            status: 200,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*',
              'Access-Control-Allow-Methods': 'POST, OPTIONS',
              'Access-Control-Allow-Headers': 'Content-Type',
            }
          }
        );
      } else {
        return NextResponse.json(
          { 
            error: "Error al generar el contenido",
            details: apiError.message || "Error desconocido"
          },
          { 
            status: 500,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*',
              'Access-Control-Allow-Methods': 'POST, OPTIONS',
              'Access-Control-Allow-Headers': 'Content-Type',
            }
          }
        );
      }
    }
  } catch (e: any) {
    console.error('Error general:', e);
    console.error('Error stack:', e.stack);
    return NextResponse.json(
      { brandKit: DUMMY_BRAND_KIT, fallback: true, error: "Error general", details: e.message },
      { 
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        }
      }
    );
  }
}  