export async function exportToPDF(content: string, filename: string) {
  try {
    // Dynamic import to reduce bundle size
    const html2pdf = await import('html2pdf.js');
    
    const element = document.createElement('div');
    element.innerHTML = `
      <div style="padding: 40px; font-family: 'Inter', Arial, sans-serif; max-width: 800px; margin: 0 auto; line-height: 1.6;">
        <div style="white-space: pre-wrap; word-wrap: break-word;">${content.replace(/\n/g, '<br>')}</div>
      </div>
    `;

    const opt = {
      margin: [10, 10, 10, 10],
      filename: `${filename}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
    };

    await html2pdf.default().set(opt).from(element).save();
    return true;
  } catch (error) {
    console.error('PDF export error:', error);
    throw new Error('Error al exportar a PDF. Por favor, instala html2pdf.js');
  }
}


