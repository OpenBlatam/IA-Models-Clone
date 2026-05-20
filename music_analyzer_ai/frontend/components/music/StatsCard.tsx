'use client';

import { ReactNode } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon?: ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}

export function StatsCard({ title, value, change, icon, trend }: StatsCardProps) {
  const getTrendIcon = () => {
    if (trend === 'up') return <TrendingUp className="w-4 h-4 text-green-400" />;
    if (trend === 'down') return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-gray-400" />;
  };

  const getTrendColor = () => {
    if (trend === 'up') return 'text-green-400';
    if (trend === 'down') return 'text-red-400';
    return 'text-gray-400';
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 hover:bg-white/15 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm text-gray-400">{title}</p>
        {icon && <div className="text-purple-300">{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2">
        <p className="text-2xl font-bold text-white">{value}</p>
        {change !== undefined && (
          <div className={`flex items-center gap-1 text-sm ${getTrendColor()}`}>
            {getTrendIcon()}
            <span>{Math.abs(change)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}

