'use client';

import React, { useState } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';
import { clsx } from 'clsx';

interface JSONViewerProps {
  data: any;
  collapsed?: boolean;
  level?: number;
  className?: string;
}

export const JSONViewer: React.FC<JSONViewerProps> = ({
  data,
  collapsed = false,
  level = 0,
  className,
}) => {
  const [isCollapsed, setIsCollapsed] = useState(collapsed);

  const renderValue = (value: any, key?: string): React.ReactNode => {
    if (value === null) {
      return <span className="text-purple-600 dark:text-purple-400">null</span>;
    }

    if (value === undefined) {
      return <span className="text-gray-400">undefined</span>;
    }

    if (typeof value === 'string') {
      return <span className="text-green-600 dark:text-green-400">"{value}"</span>;
    }

    if (typeof value === 'number') {
      return <span className="text-blue-600 dark:text-blue-400">{value}</span>;
    }

    if (typeof value === 'boolean') {
      return <span className="text-orange-600 dark:text-orange-400">{String(value)}</span>;
    }

    if (Array.isArray(value)) {
      if (value.length === 0) {
        return <span className="text-gray-500">[]</span>;
      }

      return (
        <div className="ml-4">
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="flex items-center space-x-1 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
            <span>[</span>
          </button>
          {!isCollapsed && (
            <div className="ml-4">
              {value.map((item, index) => (
                <div key={index} className="flex">
                  <span className="text-gray-500 mr-2">{index}:</span>
                  <div>{renderValue(item)}</div>
                </div>
              ))}
            </div>
          )}
          <span className="text-gray-500">]</span>
        </div>
      );
    }

    if (typeof value === 'object') {
      const keys = Object.keys(value);
      if (keys.length === 0) {
        return <span className="text-gray-500">{'{}'}</span>;
      }

      return (
        <div className="ml-4">
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="flex items-center space-x-1 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
            <span>{'{'}</span>
          </button>
          {!isCollapsed && (
            <div className="ml-4">
              {keys.map((key) => (
                <div key={key} className="flex">
                  <span className="text-purple-600 dark:text-purple-400 mr-2">
                    "{key}":
                  </span>
                  <div>{renderValue(value[key], key)}</div>
                </div>
              ))}
            </div>
          )}
          <span className="text-gray-500">{'}'}</span>
        </div>
      );
    }

    return <span>{String(value)}</span>;
  };

  return (
    <div className={clsx('font-mono text-sm', className)}>
      {renderValue(data)}
    </div>
  );
};


