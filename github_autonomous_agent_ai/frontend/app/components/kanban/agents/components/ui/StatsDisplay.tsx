"use client";

interface StatsDisplayProps {
  label: string;
  value: string | number;
  className?: string;
}

export function StatsDisplay({ label, value, className }: StatsDisplayProps) {
  return (
    <div className={`text-right ${className}`}>
      <p className="text-[11px] text-gray-500">{label}</p>
      <p className="text-xs font-semibold text-gray-900">{value}</p>
    </div>
  );
}








