'use client';

import { useRobotStore } from '@/lib/store/robotStore';
import { TrendingUp, Activity, Zap, Clock } from 'lucide-react';
import { motion } from 'framer-motion';
import AnimatedNumber from '@/components/ui/AnimatedNumber';

export default function QuickStats() {
  const { status } = useRobotStore();

  const stats = [
    {
      label: 'Estado',
      value: status?.robot_status.connected ? 'Conectado' : 'Desconectado',
      icon: <Activity className="w-5 h-5" />,
      color: status?.robot_status.connected ? 'text-green-600' : 'text-red-600',
    },
    {
      label: 'Movimiento',
      value: status?.robot_status.moving ? 'En movimiento' : 'Detenido',
      icon: <TrendingUp className="w-5 h-5" />,
      color: status?.robot_status.moving ? 'text-tesla-blue' : 'text-tesla-gray-dark',
    },
    {
      label: 'Velocidad',
      value: status?.robot_status.velocity
        ? `${Math.sqrt(
            status.robot_status.velocity[0] ** 2 +
            status.robot_status.velocity[1] ** 2 +
            status.robot_status.velocity[2] ** 2
          ).toFixed(2)} m/s`
        : '0.00 m/s',
      icon: <Zap className="w-5 h-5" />,
      color: 'text-tesla-blue',
    },
    {
      label: 'Tiempo activo',
      value: '2h 34m',
      icon: <Clock className="w-5 h-5" />,
      color: 'text-tesla-gray-dark',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-tesla-md">
      {stats.map((stat, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
          whileHover={{ y: -4, scale: 1.02 }}
          className="p-tesla-lg bg-white rounded-md border border-gray-200 shadow-sm hover:shadow-tesla-md transition-all card-hover"
        >
          <div className="flex items-center gap-tesla-sm mb-tesla-sm">
            <motion.div
              className={`${stat.color} opacity-80`}
              whileHover={{ scale: 1.1, rotate: 5 }}
              transition={{ duration: 0.2 }}
            >
              {stat.icon}
            </motion.div>
            <span className="text-xs text-tesla-gray-dark font-medium uppercase tracking-wide">{stat.label}</span>
          </div>
          <p className="text-xl font-semibold text-tesla-black">
            {stat.value.includes('m/s') || stat.value.includes('h') ? (
              stat.value
            ) : (
              <AnimatedNumber value={parseFloat(stat.value) || 0} decimals={0} />
            )}
          </p>
        </motion.div>
      ))}
    </div>
  );
}


