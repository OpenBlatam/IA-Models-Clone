/**
 * Intentar reparar JSON malformado
 */
export function tryRepairJSON(jsonString: string): string | null {
  try {
    JSON.parse(jsonString);
    return jsonString;
  } catch (e: any) {
    let repaired = jsonString;

    // Intentar cerrar strings no terminados
    if (e.message.includes('Unterminated string')) {
      const matches = repaired.match(/"/g);
      if (matches && matches.length % 2 !== 0) {
        repaired = repaired.replace(/([^\\])"([^"]*)$/, '$1"$2"');
      }
    }

    // Intentar cerrar llaves y corchetes abiertos
    const openBraces = (repaired.match(/\{/g) || []).length;
    const closeBraces = (repaired.match(/\}/g) || []).length;
    const openBrackets = (repaired.match(/\[/g) || []).length;
    const closeBrackets = (repaired.match(/\]/g) || []).length;

    for (let i = 0; i < openBraces - closeBraces; i++) {
      repaired += '}';
    }
    for (let i = 0; i < openBrackets - closeBrackets; i++) {
      repaired += ']';
    }

    try {
      JSON.parse(repaired);
      return repaired;
    } catch {
      return null;
    }
  }
}

/**
 * Extraer JSON parcial del contenido
 */
export function extractPartialJSON(content: string): string | null {
  if (!content || typeof content !== 'string') {
    return null;
  }

  // Limpiar el contenido
  let cleanContent = content.trim();
  
  // Remover markdown code blocks si existen
  cleanContent = cleanContent
    .replace(/```json\s*/g, '')
    .replace(/```\s*/g, '')
    .replace(/^json\s*/g, '')
    .trim();
  
  // Si el contenido empieza con {, asegurarse de que no haya texto antes
  const jsonStartIndex = cleanContent.indexOf('{');
  if (jsonStartIndex > 0 && jsonStartIndex < 50) {
    cleanContent = cleanContent.substring(jsonStartIndex);
  }

  // Buscar JSON completo entre llaves
  const jsonMatch = cleanContent.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    const jsonString = jsonMatch[0];
    
    // Intentar parsear directamente
    try {
      JSON.parse(jsonString);
      return jsonString;
    } catch {
      // Intentar reparar
      const repaired = tryRepairJSON(jsonString);
      if (repaired) {
        try {
          JSON.parse(repaired);
          return repaired;
        } catch {
          // Intentar extraer solo la parte válida
          for (let i = jsonString.length; i > 0; i--) {
            const substring = jsonString.substring(0, i);
            const repairedSub = tryRepairJSON(substring);
            if (repairedSub) {
              try {
                const parsed = JSON.parse(repairedSub);
                if (parsed.plan || parsed.explanation || parsed.files_to_create || parsed.files_to_modify) {
                  return repairedSub;
                }
              } catch {
                continue;
              }
            }
          }
        }
      }
    }
  }

  // Si no se encontró JSON, intentar parsear todo el contenido si empieza con {
  if (cleanContent.trim().startsWith('{')) {
    const repaired = tryRepairJSON(cleanContent);
    if (repaired) {
      try {
        JSON.parse(repaired);
        return repaired;
      } catch {
        return null;
      }
    }
  }

  return null;
}

