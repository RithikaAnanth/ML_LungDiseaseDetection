import { useEffect, useMemo, useState } from 'react'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import FeatureGraph from './components/FeatureGraph'

type ModelKey = 'svm' | 'rf' | 'knn'
type ModelSelect = 'all' | ModelKey

type Probabilities = Record<string, number>

type ModelResult =
  | {
      class_index: number
      class_name: string
      probabilities?: Probabilities
    }
  | {
      error: string
    }

type PredictResponse = {
  classes: string[]
  available_models: string[]
  results: Partial<Record<ModelKey, ModelResult>>
  features?: {
    texture?: number
    pattern?: number
    structure?: number
  }
  explanation?: string
  feature_insights?: {
    average_probabilities?: Probabilities
  }
  error?: string
  details?: string
}

const API_BASE = 'http://localhost:5000'
const MODEL_KEYS: ModelKey[] = ['svm', 'rf', 'knn']
const PIE_COLORS: Record<string, string> = {
  Normal: '#22c55e',
  Pneumonia: '#f59e0b',
  COVID: '#ef4444',
}
const PIE_LABELS = ['Normal', 'Pneumonia', 'COVID'] as const

function prettyModelName(k: string) {
  const key = k.toLowerCase()
  if (key === 'svm') return 'SVM'
  if (key === 'rf') return 'RF'
  if (key === 'knn') return 'KNN'
  return k.toUpperCase()
}

function clamp01(x: number) {
  if (!Number.isFinite(x)) return 0
  return Math.max(0, Math.min(1, x))
}

function pct(x: number) {
  return `${Math.round(clamp01(x) * 100)}%`
}

function sortProbDesc(p?: Probabilities) {
  return Object.entries(p || {}).sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
}

function argmaxLabel(p?: Probabilities) {
  const entries = sortProbDesc(p)
  return entries.length ? entries[0]![0] : null
}

function ConfidenceBars({ probabilities }: { probabilities?: Probabilities }) {
  if (!probabilities) return <div className="empty">No confidence scores returned for this model.</div>
  return (
    <div className="bars">
      {sortProbDesc(probabilities).map(([label, v]) => (
        <div className="barRow" key={label}>
          <div className="barLabel">{label}</div>
          <div className="barTrack">
            <div className="barFill" style={{ width: `${clamp01(v) * 100}%` }} />
          </div>
          <div className="barVal">{pct(v)}</div>
        </div>
      ))}
    </div>
  )
}

function ConfidencePieChart({
  probabilities,
  title = 'Confidence (pie)',
}: {
  probabilities?: Probabilities
  title?: string
}) {
  const data = useMemo(() => {
    const p = probabilities || {}
    const raw = PIE_LABELS.map((k) => ({
      name: k,
      value: clamp01(Number(p[k] ?? 0)),
    })).filter((entry) => entry.value > 0)

    const sum = raw.reduce((acc, d) => acc + d.value, 0)
    if (sum <= 0) return []

    // Preserve valid probability distributions so the dominant class stays visually dominant.
    if (sum <= 1.001) return raw

    return raw.map((d) => ({ ...d, value: d.value / sum }))
  }, [probabilities])

  if (!data.length) return <div className="empty">No confidence scores to plot.</div>

  return (
    <div className="chartWrap">
      <div className="chartTitle">{title}</div>
      <div className="chartBox">
        <div className="chartCanvas">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart margin={{ top: 8, right: 8, bottom: 28, left: 8 }}>
              <Tooltip
                formatter={(v: any, name: any) => [`${Math.round(Number(v) * 100)}%`, String(name)]}
                contentStyle={{
                  background: 'rgba(10, 15, 32, 0.92)',
                  border: '1px solid rgba(255,255,255,0.14)',
                  borderRadius: 12,
                  color: 'rgba(255,255,255,0.92)',
                }}
                itemStyle={{ color: 'rgba(255,255,255,0.92)' }}
              />
              <Pie
                data={data}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius="72%"
                innerRadius="46%"
                paddingAngle={4}
                isAnimationActive
                animationDuration={550}
                labelLine={false}
                label={({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
                  if (!percent || percent < 0.06) return null

                  const radius = Number(innerRadius) + (Number(outerRadius) - Number(innerRadius)) * 0.55
                  const x = Number(cx) + radius * Math.cos((-midAngle * Math.PI) / 180)
                  const y = Number(cy) + radius * Math.sin((-midAngle * Math.PI) / 180)

                  return (
                    <text
                      x={x}
                      y={y}
                      fill="rgba(255,255,255,0.96)"
                      textAnchor="middle"
                      dominantBaseline="central"
                      fontSize={12}
                      fontWeight={700}
                    >
                      {`${Math.round(percent * 100)}%`}
                    </text>
                  )
                }}
              >
                {data.map((entry) => (
                  <Cell key={entry.name} fill={PIE_COLORS[entry.name] || '#7c5cff'} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="chartLegend">
          {PIE_LABELS.map((label) => (
            <div className="chartLegendItem" key={label}>
              <span
                className="chartLegendDot"
                style={{ backgroundColor: PIE_COLORS[label] || '#7c5cff' }}
              />
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ModelCard({
  modelKey,
  available,
  result,
}: {
  modelKey: ModelKey
  available: string[]
  result?: ModelResult
}) {
  const title = prettyModelName(modelKey)
  const isAvail = available.includes(modelKey)

  if (!isAvail) {
    return (
      <div className="modelCard">
        <div className="modelHead">
          <div className="modelName">{title}</div>
          <div className="badge">not available</div>
        </div>
        <div className="pred">
          Add <code>{modelKey}_model.pkl</code> in <code>model/</code> to enable.
        </div>
      </div>
    )
  }

  if (!result || 'error' in result) {
    return (
      <div className="modelCard">
        <div className="modelHead">
          <div className="modelName">{title}</div>
          <div className="badge">error</div>
        </div>
        <div className="pred">{(result && 'error' in result && result.error) || 'Prediction failed.'}</div>
      </div>
    )
  }

  return (
    <div className="modelCard">
      <div className="modelHead">
        <div className="modelName">{title}</div>
        <div className="badge">ok</div>
      </div>
      <div className="pred">
        Prediction: <b>{result.class_name ?? '—'}</b>
      </div>
      <ConfidenceBars probabilities={result.probabilities} />
    </div>
  )
}

function InsightsPanel({ payload }: { payload?: PredictResponse }) {
  const avg = payload?.feature_insights?.average_probabilities
  if (!avg) {
    return <div className="empty">Run a prediction to see insights.</div>
  }
  const top = argmaxLabel(avg)
  return (
    <div className="insights">
      <div className="panel" style={{ width: '100%' }}>
        <div className="kpi">
          <div className="kpiTitle">Overall (avg across models)</div>
          <div className="kpiValue">{top || '—'}</div>
        </div>
        <ConfidenceBars probabilities={avg} />
        <div className="small" style={{ marginTop: 10 }}>
          This panel averages confidence across the available model results.
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [model, setModel] = useState<ModelSelect>('all')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<{ kind: 'muted' | 'good' | 'bad'; text: string }>({
    kind: 'muted',
    text: '',
  })
  const [payload, setPayload] = useState<PredictResponse | null>(null)

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/health`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(() => {
        if (!cancelled) setBackendOk(true)
      })
      .catch(() => {
        if (!cancelled) setBackendOk(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const availableModels = payload?.available_models || []
  const results = payload?.results || {}
  const featureData = payload?.features
  const explanation = payload?.explanation

  const pieSource = useMemo(() => {
    if (!payload) return undefined

    // If a single model is selected, prefer its own probabilities.
    if (model !== 'all') {
      const r = results[model]
      if (r && !('error' in r) && r.probabilities) return r.probabilities
    }

    // For ALL (or as fallback), use average_probabilities from backend when available.
    const avg = payload.feature_insights?.average_probabilities
    if (avg) return avg

    // Optional fallback: average over all available model probability dicts.
    const probSources: Probabilities[] = []
    for (const k of MODEL_KEYS) {
      const r = results[k]
      if (r && !('error' in r) && r.probabilities) probSources.push(r.probabilities)
    }
    if (probSources.length) {
      const labels = ['Normal', 'Pneumonia', 'COVID']
      const out: Probabilities = {}
      for (const lbl of labels) {
        let s = 0
        for (const p of probSources) s += Number(p[lbl] ?? 0)
        out[lbl] = s / probSources.length
      }
      return out
    }

    return undefined
  }, [model, payload, results])

  const backendBadge = useMemo(() => {
    if (backendOk === null) return { text: 'checking…', cls: 'muted' }
    if (backendOk) return { text: 'connected', cls: 'good' }
    return { text: 'not reachable', cls: 'bad' }
  }, [backendOk])

  async function onPredict() {
    if (!file) return
    setLoading(true)
    setStatus({ kind: 'muted', text: 'Predicting…' })
    setPayload(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)

      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      })

      const data = (await response.json()) as PredictResponse

      if (!response.ok) {
        throw new Error(data.error || data.details || 'Prediction failed.')
      }

      setPayload(data)
      setStatus({ kind: 'good', text: 'Done.' })
    } catch (e: any) {
      setStatus({ kind: 'bad', text: `Error: ${e?.message || String(e)}` })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="container">
      <header className="header">
        <div>
          <h1>Lung Disease Detection</h1>
          <p className="subtitle">Upload a chest X-ray to predict: Normal, Pneumonia, COVID</p>
        </div>
        <div className="pill">
          Backend:{' '}
          <span className={`pillValue ${backendBadge.cls}`}>{backendBadge.text}</span>
        </div>
      </header>

      <div className="main-container">
        <aside className="left-panel-column">
        <div className="left-panel">
          <section className="card">
          <h2>Upload & Predict</h2>
          <div className="stack">
            <label className="file">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              <span className="fileBtn">Upload Image</span>
              <span className="fileName">{file ? file.name : 'No file selected'}</span>
            </label>

            <div className="row">
              <label className="label" htmlFor="modelSelect">
                Select Model
              </label>
              <select
                id="modelSelect"
                className="select"
                value={model}
                onChange={(e) => setModel(e.target.value as ModelSelect)}
              >
                <option value="all">ALL</option>
                <option value="svm">SVM</option>
                <option value="rf">RF</option>
                <option value="knn">KNN</option>
              </select>
            </div>

            <button className="btn" onClick={onPredict} disabled={!file || loading}>
              {loading ? 'Predicting…' : 'Predict'}
            </button>
            <div className={`status ${status.kind}`}>{status.text}</div>
          </div>
        </section>

          <section className="card">
          <h2>Uploaded Image</h2>
          <div className="imageWrap">
            {previewUrl ? (
              <img className="preview" src={previewUrl} alt="Uploaded preview" />
            ) : (
              <div className="hint">Upload an image to preview it here.</div>
            )}
          </div>
          </section>
        </div>
        </aside>

        <main className="right-panel">
          <section className="card results-section">
          <h2>Model Results</h2>
          <div className="resultsGrid">
            {payload ? (
              MODEL_KEYS.map((k) => (
                <ModelCard
                  key={k}
                  modelKey={k}
                  available={availableModels}
                  result={results[k]}
                />
              ))
            ) : (
              <div className="empty">{loading ? 'Running prediction…' : 'Run prediction to view results.'}</div>
            )}
          </div>
        </section>

          {payload ? (
            <section className="card chart-section">
              <h2>Confidence Overview</h2>
              <ConfidencePieChart probabilities={pieSource} title="Confidence (overview)" />
            </section>
          ) : null}

          <section className="card decision-section">
          <h2>Aggregated Decision</h2>
          <InsightsPanel payload={payload || undefined} />
        </section>

          <section className="card feature-section">
            <h2>Feature Analysis</h2>
            <div className="feature-analysis-layout">
              <div className="feature-graph-pane">
                <FeatureGraph featureData={featureData} />
              </div>

              <div className="feature-explanation-pane">
                <div className="panel" style={{ width: '100%' }}>
                  <div className="kpi">
                    <div className="kpiTitle">Technical Explanation</div>
                    <div className="kpiValue">{explanation ? 'Available' : 'Pending'}</div>
                  </div>
                  <div className="small" style={{ marginTop: 10, lineHeight: 1.7 }}>
                    {explanation || 'Run prediction to view results.'}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>

      
      </div>
    </div>
  )
}
