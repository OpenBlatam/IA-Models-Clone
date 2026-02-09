/**
 * Component to display psychological profile
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Tabs } from '@/components/ui';
import { PersonalityChart } from './PersonalityChart';
import { PersonalityRadarChart } from './PersonalityRadarChart';
import type { PsychologicalProfileResponse } from '@/lib/types';
import { CheckCircle2, AlertTriangle, BarChart3, Radar } from 'lucide-react';

export interface ProfileDisplayProps {
  profile: PsychologicalProfileResponse;
}

export const ProfileDisplay: React.FC<ProfileDisplayProps> = ({ profile }) => {
  const chartTabs = [
    {
      id: 'bar',
      label: 'Gráfico de Barras',
      content: <PersonalityChart traits={profile.personality_traits} />,
    },
    {
      id: 'radar',
      label: 'Gráfico Radar',
      content: <PersonalityRadarChart traits={profile.personality_traits} />,
    },
  ];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Rasgos de Personalidad (Big Five)</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs tabs={chartTabs} defaultTab="bar" />
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Fortalezas</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {profile.strengths.map((strength, index) => (
                <li key={index} className="flex items-start gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <span>{strength}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Factores de Riesgo</CardTitle>
          </CardHeader>
          <CardContent>
            {profile.risk_factors.length > 0 ? (
              <ul className="space-y-2">
                {profile.risk_factors.map((risk, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span>{risk}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground">No se detectaron factores de riesgo</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recomendaciones</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-3">
            {profile.recommendations.map((recommendation, index) => (
              <li key={index} className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium">
                  {index + 1}
                </span>
                <span>{recommendation}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Score de Confianza</span>
            <span className="text-2xl font-bold">
              {(profile.confidence_score * 100).toFixed(1)}%
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

