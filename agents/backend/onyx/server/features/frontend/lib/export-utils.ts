export async function exportToPDF(content: string, filename: string) {
  // Dynamic import to reduce bundle size
  const html2pdf = await import('html2pdf.js').catch(() => null);
  
  if (!html2pdf) {
    throw new Error('PDF export not available. Please install html2pdf.js');
  }

  const element = document.createElement('div');
  element.innerHTML = `
    <div style="padding: 40px; font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
      <div style="white-space: pre-wrap; word-wrap: break-word;">${content}</div>
    </div>
  `;

  const opt = {
    margin: 1,
    filename: `${filename}.pdf`,
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2 },
    jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
  };

  await html2pdf.default().set(opt).from(element).save();
}

export function exportToHTML(content: string, filename: string) {
  const html = `
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${filename}</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 800px;
      margin: 0 auto;
      padding: 40px;
      line-height: 1.6;
    }
    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
    }
  </style>
</head>
<body>
  <pre>${content}</pre>
</body>
</html>
  `;

  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.html`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}


