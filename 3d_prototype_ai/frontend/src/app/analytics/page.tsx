'use client'

import { useQuery } from '@tanstack/react-query'
import { analyticsService } from '@/services/analytics.service'
import { formatCurrency } from '@/lib/utils'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

const COLORS = ['#0ea5e9', '#a855f7', '#10b981', '#f59e0b', '#ef4444']

export default function AnalyticsPage() {
  const { data: analytics, isLoading } = useQuery({
    queryKey: ['analytics'],
    queryFn: () => analyticsService.getAnalytics(),
  })

  const { data: trends } = useQuery({
    queryKey: ['trends'],
    queryFn: () => analyticsService.getTrends(30),
  })

  if (isLoading) {
    return (
      <div className="min-h-screen">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">Cargando...</div>
        </div>
      </div>
    )
  }

  const chartData = analytics?.trends?.map((trend: { date: string; count: number; total_cost: number }) => ({
    date: new Date(trend.date).toLocaleDateString('es-MX', { month: 'short', day: 'numeric' }),
    prototipos: trend.count,
    costo: trend.total_cost,
  })) || []

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Analytics</h1>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Tendencias de Prototipos</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="prototipos" stroke="#0ea5e9" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Costos por Fecha</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                <Legend />
                <Bar dataKey="costo" fill="#a855f7" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Resumen de Estadísticas</h2>
          <div className="grid md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-600">Total Prototipos</div>
              <div className="text-2xl font-bold text-gray-900">
                {analytics?.total_prototypes || 0}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Costo Total</div>
              <div className="text-2xl font-bold text-primary-600">
                {formatCurrency(analytics?.total_cost || 0)}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Costo Promedio</div>
              <div className="text-2xl font-bold text-green-600">
                {formatCurrency(analytics?.average_cost || 0)}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Tipo Más Común</div>
              <div className="text-2xl font-bold text-purple-600 capitalize">
                {analytics?.most_common_type || 'N/A'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

