/**
 * Component to display validation report with tabs
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Tabs } from '@/components/ui';
import type { ValidationReportResponse } from '@/lib/types';
import { FileText, TrendingUp, Heart, MessageSquare, Calendar } from 'lucide-react';

export interface ReportViewerProps {
  report: ValidationReportResponse;
}

export const ReportViewer: React.FC<ReportViewerProps> = ({ report }) => {
  const tabs = [
    {
      id: 'summary',
      label: 'Resumen',
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground whitespace-pre-wrap">{report.summary}</p>
        </div>
      ),
    },
    {
      id: 'insights',
      label: 'Insights',
      content: (
        <div className="space-y-4">
          {Object.entries(report.social_media_insights).map(([platform, insights]) => (
            <Card key={platform}>
              <CardHeader>
                <CardTitle className="text-lg capitalize">{platform}</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-sm text-muted-foreground whitespace-pre-wrap">
                  {JSON.stringify(insights, null, 2)}
                </pre>
              </CardContent>
            </Card>
          ))}
        </div>
      ),
    },
    {
      id: 'sentiment',
      label: 'Sentimientos',
      content: (
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <pre className="text-sm text-muted-foreground whitespace-pre-wrap">
                {JSON.stringify(report.sentiment_analysis, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </div>
      ),
    },
    {
      id: 'timeline',
      label: 'Timeline',
      content: (
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <pre className="text-sm text-muted-foreground whitespace-pre-wrap">
                {JSON.stringify(report.timeline_analysis, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </div>
      ),
    },
    {
      id: 'content',
      label: 'Contenido',
      content: (
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <pre className="text-sm text-muted-foreground whitespace-pre-wrap">
                {JSON.stringify(report.content_analysis, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </div>
      ),
    },
    {
      id: 'interactions',
      label: 'Interacciones',
      content: (
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <pre className="text-sm text-muted-foreground whitespace-pre-wrap">
                {JSON.stringify(report.interaction_patterns, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </div>
      ),
    },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Reporte de Validación</CardTitle>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Calendar className="h-4 w-4" aria-hidden="true" />
            <time dateTime={report.generated_at}>
              {new Date(report.generated_at).toLocaleDateString('es-ES')}
            </time>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs tabs={tabs} defaultTab="summary" />
      </CardContent>
    </Card>
  );
};




