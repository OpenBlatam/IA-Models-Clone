/**
 * Print utilities
 */

// Print element
export function printElement(elementId: string, options?: { title?: string }) {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return;
  }

  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`Element with id "${elementId}" not found`);
    return;
  }

  const printWindow = window.open('', '_blank');
  if (!printWindow) {
    console.error('Failed to open print window');
    return;
  }

  printWindow.document.write(`
    <html>
      <head>
        <title>${options?.title || 'Print'}</title>
        <style>
          body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
          @media print {
            body { margin: 0; padding: 0; }
          }
        </style>
      </head>
      <body>
        ${element.innerHTML}
      </body>
    </html>
  `);

  printWindow.document.close();
  printWindow.focus();
  printWindow.print();
  printWindow.close();
}

// Print page
export function printPage() {
  if (typeof window === 'undefined') {
    return;
  }
  window.print();
}

// Print URL
export function printURL(url: string) {
  if (typeof window === 'undefined') {
    return;
  }

  const printWindow = window.open(url, '_blank');
  if (!printWindow) {
    console.error('Failed to open print window');
    return;
  }

  printWindow.onload = () => {
    printWindow.print();
  };
}



