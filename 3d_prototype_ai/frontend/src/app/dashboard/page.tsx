'use client'

import { useQuery } from '@tanstack/react-query'
import { analyticsService } from '@/services/analytics.service'
import { prototypeService } from '@/services/prototype.service'
import { TrendingUp, DollarSign, Package, Activity } from 'lucide-react'
import { formatCurrency } from '@/lib/utils'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

export default function DashboardPage() {
  const { data: analytics, isLoading } = useQuery({
    queryKey: ['analytics'],
    queryFn: () => analyticsService.getAnalytics(),
  })

  const { data: history } = useQuery({
    queryKey: ['history', 1, 10],
    queryFn: () => prototypeService.getHistory(1, 10),
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
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Dashboard</h1>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-primary-100 p-3 rounded-lg">
                <Package className="w-6 h-6 text-primary-600" />
              </div>
              <TrendingUp className="w-5 h-5 text-green-500" />
            </div>
            <div className="text-2xl font-bold text-gray-900">{analytics?.total_prototypes || 0}</div>
            <div className="text-sm text-gray-600">Total Prototipos</div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-green-100 p-3 rounded-lg">
                <DollarSign className="w-6 h-6 text-green-600" />
              </div>
              <TrendingUp className="w-5 h-5 text-green-500" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {formatCurrency(analytics?.total_cost || 0)}
            </div>
            <div className="text-sm text-gray-600">Costo Total</div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-blue-100 p-3 rounded-lg">
                <DollarSign className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {formatCurrency(analytics?.average_cost || 0)}
            </div>
            <div className="text-sm text-gray-600">Costo Promedio</div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-purple-100 p-3 rounded-lg">
                <Activity className="w-6 h-6 text-purple-600" />
              </div>
            </div>
            <div className="text-2xl font-bold text-gray-900 capitalize">
              {analytics?.most_common_type || 'N/A'}
            </div>
            <div className="text-sm text-gray-600">Tipo Más Común</div>
          </div>
        </div>

        {/* Charts */}
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

        {/* Recent Prototypes */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Prototipos Recientes</h2>
          <div className="space-y-4">
            {history?.items?.map((item) => (
              <div
                key={item.id}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="font-semibold text-lg">{item.product_name}</h3>
                    <p className="text-sm text-gray-600 capitalize">{item.product_type}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-primary-600">
                      {formatCurrency(item.total_cost_estimate)}
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(item.generated_at).toLocaleDateString('es-MX')}
                    </div>
                  </div>
                </div>
              </div>
            )) || <div className="text-gray-500">No hay prototipos recientes</div>}
          </div>
        </div>
      </div>
    </div>
  )
}

