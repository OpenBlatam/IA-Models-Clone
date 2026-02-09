import { useMemo } from 'react';
import { ChartComponents, CHART_COLORS } from '@/lib/integrations/recharts';
import { cn } from '@/lib/utils';

const {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} = ChartComponents;

interface ChartData {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface BaseChartProps {
  data: ChartData[];
  className?: string;
  height?: number;
}

interface LineChartProps extends BaseChartProps {
  dataKey: string;
  strokeColor?: string;
}

interface BarChartProps extends BaseChartProps {
  dataKey: string;
  fillColor?: string;
}

interface PieChartProps extends BaseChartProps {
  dataKey: string;
  nameKey: string;
}

interface AreaChartProps extends BaseChartProps {
  dataKey: string;
  fillColor?: string;
}

export const LineChartComponent = ({
  data,
  dataKey,
  strokeColor = CHART_COLORS[0],
  className = '',
  height = 300,
}: LineChartProps): JSX.Element => {
  return (
    <div className={cn('w-full', className)}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey={dataKey} stroke={strokeColor} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export const BarChartComponent = ({
  data,
  dataKey,
  fillColor = CHART_COLORS[0],
  className = '',
  height = 300,
}: BarChartProps): JSX.Element => {
  return (
    <div className={cn('w-full', className)}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey={dataKey} fill={fillColor} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const PieChartComponent = ({
  data,
  dataKey,
  nameKey,
  className = '',
  height = 300,
}: PieChartProps): JSX.Element => {
  const colors = useMemo(() => {
    return data.map((_, index) => CHART_COLORS[index % CHART_COLORS.length]);
  }, [data]);

  return (
    <div className={cn('w-full', className)}>
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={data}
            dataKey={dataKey}
            nameKey={nameKey}
            cx="50%"
            cy="50%"
            outerRadius={80}
            label
          >
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={colors[index]} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export const AreaChartComponent = ({
  data,
  dataKey,
  fillColor = CHART_COLORS[0],
  className = '',
  height = 300,
}: AreaChartProps): JSX.Element => {
  return (
    <div className={cn('w-full', className)}>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Area type="monotone" dataKey={dataKey} stroke={fillColor} fill={fillColor} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};



