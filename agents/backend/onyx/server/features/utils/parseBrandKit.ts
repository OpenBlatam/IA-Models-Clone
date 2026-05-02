export function parseBrandKit(brandKit: string) {
  // Busca la sección de colores y fuentes
  const colorRegex = /#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})/g;
  const fontRegex = /(?:fuentes?|font(?:s)?):?\s*([\w\s,\-']+)/i;

  const colors = Array.from(brandKit.matchAll(colorRegex)).map(m => m[0]);
  const fontMatch = brandKit.match(fontRegex);
  let fonts = { title: '', subtitle: '', body: '' };

  if (fontMatch && fontMatch[1]) {
    // Intenta separar por comas o saltos de línea
    const fontList = fontMatch[1].split(/,|\n/).map(f => f.trim()).filter(Boolean);
    fonts = {
      title: fontList[0] || '',
      subtitle: fontList[1] || '',
      body: fontList[2] || ''
    };
  }

  // Si todo está vacío, rellena con valores por defecto
  const isEmpty = (!colors.length) && (!fonts.title && !fonts.subtitle && !fonts.body);
  if (isEmpty) {
    return {
      colors: ['#7b61ff', '#fbbf24', '#22c55e'],
      fonts: {
        title: 'Montserrat',
        subtitle: 'Roboto',
        body: 'Open Sans'
      }
    };
  }

  return { colors, fonts };
} 