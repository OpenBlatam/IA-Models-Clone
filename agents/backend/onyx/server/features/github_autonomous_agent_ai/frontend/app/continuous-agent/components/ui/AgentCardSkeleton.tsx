"use client";

import { motion } from "framer-motion";

/**
 * Skeleton loader for AgentCard component
 * 
 * Provides visual feedback while agents are loading
 */
export const AgentCardSkeleton = (): JSX.Element => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-card border rounded-lg p-6 animate-pulse"
    >
      <div className="flex justify-between items-start mb-4">
        <div className="flex-1">
          <div className="h-6 bg-gray-200 rounded w-3/4 mb-2" />
          <div className="h-4 bg-gray-200 rounded w-full" />
        </div>
        <div className="ml-4">
          <div className="w-12 h-6 bg-gray-200 rounded-full" />
        </div>
      </div>

      <div className="space-y-2 mb-4">
        <div className="h-4 bg-gray-200 rounded w-2/3" />
        <div className="h-4 bg-gray-200 rounded w-1/2" />
      </div>

      <div className="h-9 bg-gray-200 rounded w-full" />
    </motion.div>
  );
};

/**
 * Grid of skeleton loaders
 */
export const AgentCardSkeletonGrid = ({ count = 6 }: { count?: number }): JSX.Element => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {Array.from({ length: count }).map((_, index) => (
        <AgentCardSkeleton key={index} />
      ))}
    </div>
  );
};




