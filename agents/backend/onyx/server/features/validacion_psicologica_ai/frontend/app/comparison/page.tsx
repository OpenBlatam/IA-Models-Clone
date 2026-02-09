/**
 * Comparison page for validations
 */

'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui';
import { ComparisonView } from '@/components/features/ComparisonView';

export default function ComparisonPage() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground"
            aria-label="Volver al inicio"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            <span>Volver</span>
          </Link>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Comparar Validaciones</h1>
          <p className="text-muted-foreground">
            Selecciona y compara hasta 3 validaciones para analizar diferencias
          </p>
        </div>

        <ComparisonView />
      </main>
    </div>
  );
}




