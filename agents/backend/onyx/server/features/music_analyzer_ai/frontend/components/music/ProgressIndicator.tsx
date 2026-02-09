'use client';

import { useEffect, useState } from 'react';
import { CheckCircle, Loader2, XCircle } from 'lucide-react';

interface ProgressStep {
  id: string;
  label: string;
  status: 'pending' | 'loading' | 'completed' | 'error';
}

interface ProgressIndicatorProps {
  steps: ProgressStep[];
  currentStep?: string;
}

export function ProgressIndicator({ steps, currentStep }: ProgressIndicatorProps) {
  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <h3 className="text-lg font-semibold text-white mb-4">Progreso del Análisis</h3>
      <div className="space-y-4">
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-center gap-4">
            <div className="flex-shrink-0">
              {step.status === 'completed' && (
                <CheckCircle className="w-6 h-6 text-green-400" />
              )}
              {step.status === 'loading' && (
                <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
              )}
              {step.status === 'error' && (
                <XCircle className="w-6 h-6 text-red-400" />
              )}
              {step.status === 'pending' && (
                <div className="w-6 h-6 rounded-full border-2 border-gray-400" />
              )}
            </div>
            <div className="flex-1">
              <p className={`text-sm font-medium ${
                step.status === 'completed' ? 'text-green-300' :
                step.status === 'loading' ? 'text-purple-300' :
                step.status === 'error' ? 'text-red-300' :
                'text-gray-400'
              }`}>
                {step.label}
              </p>
              {step.status === 'loading' && (
                <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
                  <div className="bg-purple-400 h-1 rounded-full animate-pulse" style={{ width: '60%' }} />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

