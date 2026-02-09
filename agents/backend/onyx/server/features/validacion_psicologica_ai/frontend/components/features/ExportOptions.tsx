/**
 * Export options component
 */

'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, Button, RadioGroup } from '@/components/ui';
import { Download, FileText, FileJson, FileSpreadsheet } from 'lucide-react';

export interface ExportOptionsProps {
  onExport: (format: 'json' | 'csv' | 'pdf') => void;
  className?: string;
}

export const ExportOptions: React.FC<ExportOptionsProps> = ({
  onExport,
  className,
}) => {
  const [selectedFormat, setSelectedFormat] = useState<'json' | 'csv' | 'pdf'>('json');

  const options = [
    {
      value: 'json',
      label: 'JSON',
      description: 'Formato estructurado para desarrollo',
      icon: <FileJson className="h-4 w-4" aria-hidden="true" />,
    },
    {
      value: 'csv',
      label: 'CSV',
      description: 'Formato de hoja de cálculo',
      icon: <FileSpreadsheet className="h-4 w-4" aria-hidden="true" />,
    },
    {
      value: 'pdf',
      label: 'PDF',
      description: 'Documento imprimible',
      icon: <FileText className="h-4 w-4" aria-hidden="true" />,
    },
  ];

  const handleExport = () => {
    onExport(selectedFormat);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="h-5 w-5" aria-hidden="true" />
          Exportar Datos
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <RadioGroup
          options={options}
          value={selectedFormat}
          onChange={(value) => setSelectedFormat(value as 'json' | 'csv' | 'pdf')}
          name="export-format"
        />

        <Button
          onClick={handleExport}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              handleExport();
            }
          }}
          className="w-full"
          aria-label={`Exportar como ${selectedFormat.toUpperCase()}`}
          tabIndex={0}
        >
          <Download className="h-4 w-4 mr-2" aria-hidden="true" />
          Exportar como {selectedFormat.toUpperCase()}
        </Button>
      </CardContent>
    </Card>
  );
};



