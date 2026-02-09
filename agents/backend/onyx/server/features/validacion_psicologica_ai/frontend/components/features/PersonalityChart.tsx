/**
 * Component to display personality traits chart
 */

'use client';

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { PersonalityTraits } from '@/lib/types';

export interface PersonalityChartProps {
  traits: PersonalityTraits;
}

export const PersonalityChart: React.FC<PersonalityChartProps> = ({ traits }) => {
  const data = [
    { name: 'Apertura', value: traits.openness },
    { name: 'Responsabilidad', value: traits.conscientiousness },
    { name: 'Extraversión', value: traits.extraversion },
    { name: 'Amabilidad', value: traits.agreeableness },
    { name: 'Neuroticismo', value: traits.neuroticism },
  ];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Bar dataKey="value" fill="hsl(var(--primary))" />
      </BarChart>
    </ResponsiveContainer>
  );
};




