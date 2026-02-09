'use client';

import React, { memo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Calendar } from 'lucide-react';

interface Product {
  name: string;
  category: string;
  description: string;
  priority: number;
  key_ingredients?: string[];
  usage_frequency?: string;
}

interface RoutineSectionProps {
  title: string;
  icon: React.ReactNode;
  products: Product[];
}

export const RoutineSection: React.FC<RoutineSectionProps> = memo(({
  title,
  icon,
  products,
}) => {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center space-x-2">
          {icon}
          <CardTitle>{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {products.map((product, index) => (
            <div
              key={index}
              className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border-l-4 border-primary-500"
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">{product.name}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">{product.category}</p>
                </div>
                <span className="px-2 py-1 bg-primary-100 dark:bg-primary-900/50 text-primary-800 dark:text-primary-200 text-xs font-medium rounded">
                  Priority {product.priority}
                </span>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">{product.description}</p>
              {product.key_ingredients && product.key_ingredients.length > 0 && (
                <div className="mb-2">
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                    Key ingredients:
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {product.key_ingredients.map((ingredient, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-xs rounded border border-gray-200 dark:border-gray-700"
                      >
                        {ingredient}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {product.usage_frequency && (
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  <Calendar className="inline h-3 w-3 mr-1" />
                  {product.usage_frequency}
                </p>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
});

RoutineSection.displayName = 'RoutineSection';



