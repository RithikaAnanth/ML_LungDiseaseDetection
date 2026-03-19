import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LabelList,
  ResponsiveContainer,
  Cell,
} from 'recharts'

export type FeatureGraphProps = {
  featureData?: {
    texture?: number
    pattern?: number
    structure?: number
  }
}

type ChartDatum = {
  key: string
  label: string
  value: number
  fill: string
}

function buildChartData(featureData?: FeatureGraphProps['featureData']): ChartDatum[] {
  if (!featureData) return []

  const base = [
    {
      key: 'texture',
      label: 'Texture (GLCM)',
      value: featureData.texture ?? 0,
    },
    {
      key: 'pattern',
      label: 'Pattern (LBP)',
      value: featureData.pattern ?? 0,
    },
    {
      key: 'structure',
      label: 'Structure (HOG)',
      value: featureData.structure ?? 0,
    },
  ]

  return base.map((item) => {
    const v = typeof item.value === 'number' && Number.isFinite(item.value) ? item.value : 0
    const value = Math.max(0, Math.min(1, v))

    let fill = '#43a047' // green: low
    if (value > 0.7) fill = '#e53935' // red: high
    else if (value >= 0.4) fill = '#fdd835' // yellow: medium

    return { ...item, value, fill }
  })
}

export function FeatureGraph({ featureData }: FeatureGraphProps) {
  const data = buildChartData(featureData)

  return (
    <div style={{ width: '100%', maxWidth: '100%', margin: '0' }}>
      <div
        style={{
          padding: 16,
          border: '1px solid #e5e7eb',
          borderRadius: 4,
          backgroundColor: '#f9fafb',
          width: '100%',
          overflow: 'hidden',
        }}
      >
       

        <div style={{ width: '100%', height: 260 }}>
          {data.length ? (
            <ResponsiveContainer>
              <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="label"
                  angle={-20}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 11 }}
                />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} tickCount={6} />
                <Tooltip
                  formatter={(value: any) =>
                    typeof value === 'number' ? value.toFixed(2) : String(value)
                  }
                  labelFormatter={(label: any) => String(label)}
                  contentStyle={{
                    fontSize: 11,
                  }}
                />
                <Bar dataKey="value">
                  <LabelList
                    dataKey="value"
                    position="top"
                    formatter={(value: any) =>
                      typeof value === 'number' ? value.toFixed(2) : String(value)
                    }
                    style={{ fontSize: 11 }}
                  />
                  {data.map((entry) => (
                    <Cell key={entry.key} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div
              style={{
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 12,
                color: '#6b7280',
              }}
            >
              No feature data to visualize yet.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default FeatureGraph
