/**
 * Recent activity component with timeline
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Avatar } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { formatDistanceToNow } from 'date-fns';
import { CheckCircle2, Play, XCircle, Plus, FileText } from 'lucide-react';
import { useMemo } from 'react';

export const RecentActivity: React.FC = () => {
  const { data: validations } = useValidations();

  const activities = useMemo(() => {
    if (!validations) {
      return [];
    }

    const items: Array<{
      id: string;
      type: 'created' | 'completed' | 'failed' | 'running';
      message: string;
      timestamp: Date;
      validationId: string;
    }> = [];

    validations.forEach((validation) => {
      items.push({
        id: `${validation.id}-created`,
        type: 'created',
        message: `Validación ${validation.id.slice(0, 8)} creada`,
        timestamp: new Date(validation.created_at),
        validationId: validation.id,
      });

      if (validation.status === 'completed' && validation.completed_at) {
        items.push({
          id: `${validation.id}-completed`,
          type: 'completed',
          message: `Validación ${validation.id.slice(0, 8)} completada`,
          timestamp: new Date(validation.completed_at),
          validationId: validation.id,
        });
      } else if (validation.status === 'failed') {
        items.push({
          id: `${validation.id}-failed`,
          type: 'failed',
          message: `Validación ${validation.id.slice(0, 8)} falló`,
          timestamp: new Date(validation.updated_at),
          validationId: validation.id,
        });
      } else if (validation.status === 'running') {
        items.push({
          id: `${validation.id}-running`,
          type: 'running',
          message: `Validación ${validation.id.slice(0, 8)} en proceso`,
          timestamp: new Date(validation.updated_at),
          validationId: validation.id,
        });
      }
    });

    return items.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, 5);
  }, [validations]);

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'created':
        return <Plus className="h-4 w-4" aria-hidden="true" />;
      case 'completed':
        return <CheckCircle2 className="h-4 w-4" aria-hidden="true" />;
      case 'failed':
        return <XCircle className="h-4 w-4" aria-hidden="true" />;
      case 'running':
        return <Play className="h-4 w-4" aria-hidden="true" />;
      default:
        return <FileText className="h-4 w-4" aria-hidden="true" />;
    }
  };

  const getActivityColor = (type: string) => {
    switch (type) {
      case 'created':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'running':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  if (!activities || activities.length === 0) {
    return (
      <Card>
        <CardContent className="py-12">
          <p className="text-center text-muted-foreground">No hay actividad reciente</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Actividad Reciente</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {activities.map((activity, index) => (
            <div key={activity.id} className="flex items-start gap-3">
              <Avatar
                size="sm"
                className={getActivityColor(activity.type)}
                fallback={activity.type[0].toUpperCase()}
              >
                {getActivityIcon(activity.type)}
              </Avatar>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium">{activity.message}</p>
                <p className="text-xs text-muted-foreground">
                  {formatDistanceToNow(activity.timestamp, { addSuffix: true })}
                </p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};



