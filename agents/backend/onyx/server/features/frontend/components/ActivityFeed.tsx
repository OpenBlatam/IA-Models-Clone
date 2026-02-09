'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiActivity, FiCheckCircle, FiXCircle, FiClock, FiFileText } from 'react-icons/fi';
import { format } from 'date-fns';

interface Activity {
  id: string;
  type: 'task_created' | 'task_completed' | 'task_failed' | 'document_viewed';
  message: string;
  timestamp: Date;
  taskId?: string;
}

export default function ActivityFeed() {
  const [activities, setActivities] = useState<Activity[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem('bul_activities');
    if (stored) {
      setActivities(JSON.parse(stored).map((a: any) => ({
        ...a,
        timestamp: new Date(a.timestamp),
      })));
    }

    // Listen for new activities
    const handleActivity = (event: CustomEvent) => {
      const activity: Activity = event.detail;
      const updated = [activity, ...activities].slice(0, 50);
      setActivities(updated);
      localStorage.setItem('bul_activities', JSON.stringify(updated));
    };

    window.addEventListener('bul_activity' as any, handleActivity);
    return () => window.removeEventListener('bul_activity' as any, handleActivity);
  }, [activities]);

  const getIcon = (type: Activity['type']) => {
    switch (type) {
      case 'task_created':
        return <FiFileText className="text-blue-500" size={18} />;
      case 'task_completed':
        return <FiCheckCircle className="text-green-500" size={18} />;
      case 'task_failed':
        return <FiXCircle className="text-red-500" size={18} />;
      case 'document_viewed':
        return <FiClock className="text-yellow-500" size={18} />;
    }
  };

  if (activities.length === 0) return null;

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <FiActivity size={20} className="text-primary-600" />
        <h3 className="font-semibold text-gray-900 dark:text-white">Actividad Reciente</h3>
      </div>
      <div className="space-y-3">
        {activities.slice(0, 5).map((activity, index) => (
          <motion.div
            key={activity.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="flex items-start gap-3 p-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg"
          >
            {getIcon(activity.type)}
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-700 dark:text-gray-300">{activity.message}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {format(activity.timestamp, 'PPp')}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}


