import React, { useEffect, useMemo, useState } from 'react'

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
  feature_insights?: {
    average_probabilities?: Probabilities
  }
  error?: string
  details?: string
}

const API_BASE = 'http://localhost:5000'
const MODEL_KEYS: ModelKey[] = ['svm', 'rf', 'knn']

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
      <div className="panel">
        <div className="kpi">
          <div className="kpiTitle">Overall (avg across models)</div>
          <div className="kpiValue">{top || '—'}</div>
        </div>
        <ConfidenceBars probabilities={avg} />
        <div className="small" style={{ marginTop: 10 }}>
          This panel averages confidence across the available model results.
        </div>
      </div>
      <div className="panel">
        <div className="kpi">
          <div className="kpiTitle">Available models</div>
          <div className="kpiValue">
            {(payload?.available_models || []).map(prettyModelName).join(', ') || '—'}
          </div>
        </div>
        <div className="small">Choose a single model or compare with ALL.</div>
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
      const fd = new FormData()
      fd.append('file', file)
      fd.append('model', model)

      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd })
      const data = (await res.json()) as PredictResponse
      if (!res.ok) throw new Error(data?.error || 'Request failed')
      setPayload(data)
      setStatus({ kind: 'good', text: 'Done.' })
    } catch (e: any) {
      setStatus({ kind: 'bad', text: `Error: ${e?.message || String(e)}` })
    } finally {
      setLoading(false)
    }
  }

  return (
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

      <main className="grid">
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

        <section className="card span2">
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
              <div className="empty">{loading ? 'Running prediction…' : 'No predictions yet.'}</div>
            )}
          </div>
        </section>

        <section className="card span2">
          <h2>Feature Insights Panel (COVID / Pneumonia / Normal)</h2>
          <InsightsPanel payload={payload || undefined} />
        </section>
      </main>

      <footer className="footer">
        <div className="small">
          Tip: RF/KNN cards will show “not available” until you add <code>rf_model.pkl</code> and{' '}
          <code>knn_model.pkl</code> in <code>model/</code>.
        </div>
      </footer>
    </div>
  )
}

