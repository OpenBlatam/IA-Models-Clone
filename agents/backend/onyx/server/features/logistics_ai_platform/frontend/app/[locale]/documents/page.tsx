'use client';

import { useQuery } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import Navbar from '@/components/layout/navbar';
import { documentsApi } from '@/lib/api/documents';
import { formatDate } from '@/lib/utils';
import { Download } from 'lucide-react';

const DocumentsPage = () => {
  const t = useTranslations('documents');

  const { data: documents = [], isLoading, error } = useQuery({
    queryKey: ['documents'],
    queryFn: async () => {
      try {
        return [];
      } catch (err) {
        console.error('Error loading documents:', err);
        return [];
      }
    },
  });

  const handleDownload = (documentUrl: string) => {
    window.open(documentUrl, '_blank', 'noopener,noreferrer');
  };

  const handleKeyDown = (e: React.KeyboardEvent, documentUrl: string) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleDownload(documentUrl);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{t('title')}</h1>
            <p className="text-muted-foreground">Manage documents</p>
          </div>
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">Loading...</p>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{t('title')}</h1>
            <p className="text-muted-foreground">Manage documents</p>
          </div>
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-destructive">Error loading documents. Please try again later.</p>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">{t('title')}</h1>
          <p className="text-muted-foreground">Manage documents</p>
        </div>

        {documents.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">No documents found.</p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {documents.map((document) => (
              <Card key={document.document_id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{document.file_name}</CardTitle>
                      <CardDescription>
                        {t('documentType')}: {document.document_type} • {t('uploadedAt')}:{' '}
                        {formatDate(document.uploaded_at)}
                      </CardDescription>
                    </div>
                    <Button
                      onClick={() => handleDownload(document.file_url)}
                      onKeyDown={(e) => handleKeyDown(e, document.file_url)}
                      variant="outline"
                      aria-label={`Download ${document.file_name}`}
                      tabIndex={0}
                    >
                      <Download className="mr-2 h-4 w-4" />
                      {t('download')}
                    </Button>
                  </div>
                </CardHeader>
                {document.description && (
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{document.description}</p>
                  </CardContent>
                )}
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default DocumentsPage;
