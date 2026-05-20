'use client';

import React, { memo, useMemo } from 'react';
import { Recommendations } from '@/lib/types/api';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Sparkles, Sun, Moon, Calendar } from 'lucide-react';
import { RoutineSection } from './RoutineSection';

interface RecommendationsDisplayProps {
  recommendations: Recommendations;
}

export const RecommendationsDisplay: React.FC<RecommendationsDisplayProps> = memo(({
  recommendations,
}) => {
  const { routine, specific_recommendations, tips, priority_areas } = recommendations;

  const morningIcon = useMemo(() => <Sun className="h-5 w-5 text-yellow-500" />, []);
  const eveningIcon = useMemo(() => <Moon className="h-5 w-5 text-blue-500" />, []);
  const weeklyIcon = useMemo(() => <Calendar className="h-5 w-5 text-purple-500" />, []);

  return (
    <div className="space-y-6">
      {/* Routine */}
      <div className="space-y-4">
        <RoutineSection
          title="Morning Routine"
          icon={morningIcon}
          products={routine.morning}
        />
        <RoutineSection
          title="Evening Routine"
          icon={eveningIcon}
          products={routine.evening}
        />
        {routine.weekly && routine.weekly.length > 0 && (
          <RoutineSection
            title="Weekly Treatments"
            icon={weeklyIcon}
            products={routine.weekly}
          />
        )}
      </div>

      {/* Specific Recommendations */}
      {specific_recommendations && specific_recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <Sparkles className="h-5 w-5 text-primary-600" />
              <CardTitle>Specific Recommendations</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {specific_recommendations.map((rec, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <span className="text-primary-600 mt-1">•</span>
                  <span className="text-gray-700">{rec}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Tips */}
      {tips && tips.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Personalized Tips</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {tips.map((tip, index) => (
                <div
                  key={index}
                  className="p-3 bg-blue-50 border-l-4 border-blue-500 rounded"
                >
                  <p className="text-gray-700">{tip}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Priority Areas */}
      {priority_areas && priority_areas.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Focus Areas</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {priority_areas.map((area, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-primary-100 text-primary-800 rounded-full text-sm font-medium"
                >
                  {area}
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
});

RecommendationsDisplay.displayName = 'RecommendationsDisplay';

