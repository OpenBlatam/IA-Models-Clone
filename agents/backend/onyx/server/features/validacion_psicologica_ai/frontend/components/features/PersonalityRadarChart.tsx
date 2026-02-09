/**
 * Radar chart for personality traits
 */

'use client';

import React from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';
import type { PersonalityTraits } from '@/lib/types';

export interface PersonalityRadarChartProps {
  traits: PersonalityTraits;
}

export const PersonalityRadarChart: React.FC<PersonalityRadarChartProps> = ({ traits }) => {
  const data = [
    {
      trait: 'Apertura',
      value: traits.openness,
      fullMark: 1,
    },
    {
      trait: 'Responsabilidad',
      value: traits.conscientiousness,
      fullMark: 1,
    },
    {
      trait: 'Extraversión',
      value: traits.extraversion,
      fullMark: 1,
    },
    {
      trait: 'Amabilidad',
      value: traits.agreeableness,
      fullMark: 1,
    },
    {
      trait: 'Neuroticismo',
      value: traits.neuroticism,
      fullMark: 1,
    },
  ];

  return (
    <ResponsiveContainer width="100%" height={400}>
      <RadarChart data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="trait" />
        <PolarRadiusAxis angle={90} domain={[0, 1]} />
        <Radar
          name="Rasgos"
          dataKey="value"
          stroke="hsl(var(--primary))"
          fill="hsl(var(--primary))"
          fillOpacity={0.6}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
};




