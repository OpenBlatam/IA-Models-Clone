import React, { useMemo } from 'react';
import { View, Dimensions } from 'react-native';
import { VictoryChart, VictoryLine, VictoryBar, VictoryArea, VictoryPie } from 'victory-native';
import { useColors } from '@/theme/colors';

interface ChartData {
  x: string | number;
  y: number;
}

interface BaseChartProps {
  data: ChartData[];
  width?: number;
  height?: number;
  color?: string;
}

interface LineChartProps extends BaseChartProps {
  type: 'line';
}

interface BarChartProps extends BaseChartProps {
  type: 'bar';
}

interface AreaChartProps extends BaseChartProps {
  type: 'area';
}

interface PieChartProps {
  type: 'pie';
  data: Array<{ x: string; y: number }>;
  width?: number;
  height?: number;
}

type ChartProps = LineChartProps | BarChartProps | AreaChartProps | PieChartProps;

export function Chart(props: ChartProps): JSX.Element {
  const colors = useColors();
  const screenWidth = Dimensions.get('window').width;

  const chartWidth = useMemo(
    () => props.width || screenWidth - 32,
    [props.width, screenWidth]
  );

  const chartHeight = useMemo(() => props.height || 200, [props.height]);

  const chartColor = useMemo(
    () => props.color || colors.primary,
    [props.color, colors.primary]
  );

  if (props.type === 'pie') {
    return (
      <View style={{ alignItems: 'center' }}>
        <VictoryPie
          data={props.data}
          width={chartWidth}
          height={chartHeight}
          colorScale={[colors.primary, colors.secondary, colors.success, colors.warning]}
          innerRadius={50}
          labelRadius={({ innerRadius }) => (innerRadius || 0) + 20}
        />
      </View>
    );
  }

  const commonProps = {
    width: chartWidth,
    height: chartHeight,
    padding: { left: 50, bottom: 50, right: 20, top: 20 },
    style: {
      data: { fill: chartColor },
    },
  };

  return (
    <View style={{ alignItems: 'center' }}>
      <VictoryChart {...commonProps}>
        {props.type === 'line' && (
          <VictoryLine data={props.data} style={{ data: { stroke: chartColor } }} />
        )}
        {props.type === 'bar' && <VictoryBar data={props.data} />}
        {props.type === 'area' && (
          <VictoryArea data={props.data} style={{ data: { fill: chartColor } }} />
        )}
      </VictoryChart>
    </View>
  );
}

