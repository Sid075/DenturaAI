import React, { useState, useEffect, useMemo } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  FileCode,
  FileSpreadsheet,
  Layers,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  Zap
} from 'lucide-react'

interface ModelInfo {
  id: string
  name: string
  displayName: string
  badge: string
  type: string
  parameters: string
  accuracy: number
  f1Score: number
  isChampion: boolean
  hasWeights: boolean
  checkpoint: string
  description: string
}

interface OverallMetrics {
  totalSamples: number
  correctPredictions: number
  incorrectPredictions: number
  accuracy: number
  balancedAccuracy: number
  macroPrecision: number
  macroRecall: number
  macroF1: number
  weightedPrecision: number
  weightedRecall: number
  weightedF1: number
}

interface Metadata {
  modelName: string
  modelType?: string
  parameterCount?: string
  checkpointPath: string
  checkpointSha256: string
  datasetIdentifier: string
  evaluationTimestamp: string
  inputResolution: string
  preprocessing: string
  zeroDivisionHandling: string
  classOrder: string[]
  isChampion?: boolean
  hasWeights?: boolean
}

interface IntegrityAudit {
  status: string
  hasLeakage: boolean
  totalTestAudited: number
  duplicateCount: number
  leakageRate: number
  crossLabelDuplicateCount: number
  summary: string
  sampleCrossLabel: Array<{
    testClass: string
    trainClass: string
    testFile: string
    trainFile: string
  }>
}

interface ClassMetric {
  classIndex: number
  className: string
  displayName: string
  support: number
  truePositives: number
  falsePositives: number
  falseNegatives: number
  precision: number
  recall: number
  f1Score: number
}

interface ConfusedPair {
  trueClass: string
  predictedClass: string
  count: number
  percentageOfTrue: number
}

interface MisclassifiedSample {
  id: string
  relativePath: string
  filename: string
  trueClass: string
  predictedClass: string
  predictedConfidence: number
  trueConfidence: number
  probabilities: Record<string, number>
}

export const EvaluationDashboard: React.FC = () => {
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string>('efficientnet_b0')

  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [overallMetrics, setOverallMetrics] = useState<OverallMetrics | null>(null)
  const [integrityAudit, setIntegrityAudit] = useState<IntegrityAudit | null>(null)
  const [scientificDisclaimer, setScientificDisclaimer] = useState<string | null>(null)

  const [classLabels, setClassLabels] = useState<string[]>([])
  const [rawMatrix, setRawMatrix] = useState<number[][]>([])
  const [normalizedMatrix, setNormalizedMatrix] = useState<number[][]>([])
  const [topConfusedPairs, setTopConfusedPairs] = useState<ConfusedPair[]>([])

  const [classMetrics, setClassMetrics] = useState<ClassMetric[]>([])

  const [misclassifiedSamples, setMisclassifiedSamples] = useState<MisclassifiedSample[]>([])
  const [totalMisclassified, setTotalMisclassified] = useState<number>(0)
  const [trueClassFilter, setTrueClassFilter] = useState<string>('all')
  const [predClassFilter, setPredClassFilter] = useState<string>('all')
  const [page, setPage] = useState<number>(1)
  const pageSize = 12

  const [matrixView, setMatrixView] = useState<'raw' | 'normalized'>('raw')
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null)

  const [sortColumn, setSortColumn] = useState<keyof ClassMetric>('f1Score')
  const [sortAsc, setSortAsc] = useState<boolean>(false)

  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // 1. Fetch available models on mount
  useEffect(() => {
    fetch('/api/evaluation/models')
      .then(res => res.json())
      .then((models: ModelInfo[]) => {
        if (Array.isArray(models)) setAvailableModels(models)
      })
      .catch(err => console.error('Error fetching model registry:', err))
  }, [])

  // 2. Fetch evaluation data whenever selectedModelId changes
  const fetchEvaluationData = async (modelId: string = selectedModelId) => {
    setIsLoading(true)
    setError(null)
    try {
      const [sumRes, cmRes, classRes] = await Promise.all([
        fetch(`/api/evaluation/summary?model=${encodeURIComponent(modelId)}`),
        fetch(`/api/evaluation/confusion-matrix?model=${encodeURIComponent(modelId)}`),
        fetch(`/api/evaluation/class-metrics?model=${encodeURIComponent(modelId)}`)
      ])

      if (!sumRes.ok || !cmRes.ok || !classRes.ok) {
        throw new Error(`Failed to retrieve evaluation endpoints for model '${modelId}'.`)
      }

      const sumData = await sumRes.json()
      const cmData = await cmRes.json()
      const classData = await classRes.json()

      setMetadata(sumData.metadata)
      setOverallMetrics(sumData.overallMetrics)
      setIntegrityAudit(sumData.integrityAudit)
      setScientificDisclaimer(sumData.scientificDisclaimer)

      setClassLabels(cmData.confusionMatrix.classLabels)
      setRawMatrix(cmData.confusionMatrix.rawMatrix)
      setNormalizedMatrix(cmData.confusionMatrix.normalizedMatrix)
      setTopConfusedPairs(cmData.confusionMatrix.topConfusedPairs)

      setClassMetrics(classData.classMetrics)
      setSelectedCell(null)
    } catch (err: any) {
      setError(err.message || 'Error loading model evaluation metrics.')
    } finally {
      setIsLoading(false)
    }
  }

  // 3. Fetch misclassified samples (for models with local inference)
  const fetchMisclassified = async (modelId: string = selectedModelId) => {
    try {
      const params = new URLSearchParams({
        model: modelId,
        true_class: trueClassFilter,
        predicted_class: predClassFilter,
        limit: pageSize.toString(),
        offset: ((page - 1) * pageSize).toString()
      })
      const res = await fetch(`/api/evaluation/misclassified?${params}`)
      if (!res.ok) throw new Error('Failed to load misclassified samples.')
      const data = await res.json()
      setMisclassifiedSamples(data.samples || [])
      setTotalMisclassified(data.total || 0)
    } catch (err) {
      console.error(err)
    }
  }

  useEffect(() => {
    fetchEvaluationData(selectedModelId)
  }, [selectedModelId])

  useEffect(() => {
    if (!isLoading && !error) {
      fetchMisclassified(selectedModelId)
    }
  }, [selectedModelId, trueClassFilter, predClassFilter, page, isLoading])

  const handleModelChange = (modelId: string) => {
    if (modelId === selectedModelId) return
    setSelectedModelId(modelId)
    setPage(1)
    setTrueClassFilter('all')
    setPredClassFilter('all')
  }

  const handleRecalculate = async () => {
    setIsRefreshing(true)
    try {
      const res = await fetch('/api/evaluation/run', { method: 'POST' })
      if (!res.ok) throw new Error('Recalculation failed.')
      await fetchEvaluationData(selectedModelId)
      await fetchMisclassified(selectedModelId)
    } catch (err: any) {
      alert(`Error refreshing evaluation: ${err.message}`)
    } finally {
      setIsRefreshing(false)
    }
  }

  // Sorted class metrics
  const sortedClassMetrics = useMemo(() => {
    return [...classMetrics].sort((a, b) => {
      const valA = a[sortColumn]
      const valB = b[sortColumn]
      if (typeof valA === 'string') {
        return sortAsc
          ? (valA as string).localeCompare(valB as string)
          : (valB as string).localeCompare(valA as string)
      }
      return sortAsc ? (valA as number) - (valB as number) : (valB as number) - (valA as number)
    })
  }, [classMetrics, sortColumn, sortAsc])

  const handleSort = (column: keyof ClassMetric) => {
    if (sortColumn === column) {
      setSortAsc(!sortAsc)
    } else {
      setSortColumn(column)
      setSortAsc(false)
    }
  }

  if (isLoading && !metadata) {
    return (
      <div className="container" style={{ padding: '6rem 0', textAlign: 'center' }}>
        <div className="spinner" style={{ width: 44, height: 44, borderTopColor: '#0D9488', margin: '0 auto 1.5rem' }} />
        <h3 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#0A192F' }}>Loading Model Evaluation</h3>
        <p style={{ color: '#64748B', maxWidth: 460, margin: '0.5rem auto 0' }}>
          Loading performance benchmark metrics, confusion matrix, and class statistics...
        </p>
      </div>
    )
  }

  if (error || !overallMetrics || !metadata) {
    return (
      <div className="container" style={{ padding: '5rem 0' }}>
        <div style={{ background: '#FEF2F2', border: '1px solid #FECACA', borderRadius: 16, padding: '2.5rem', textAlign: 'center' }}>
          <AlertTriangle size={48} color="#EF4444" style={{ margin: '0 auto 1rem' }} />
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#991B1B' }}>Evaluation Data Unavailable</h2>
          <p style={{ color: '#7F1D1D', maxWidth: 560, margin: '0.75rem auto 1.5rem' }}>{error}</p>
          <button className="btn-primary" onClick={() => fetchEvaluationData(selectedModelId)}>
            <RefreshCw size={16} /> Retry Evaluation
          </button>
        </div>
      </div>
    )
  }

  const totalPages = Math.ceil(totalMisclassified / pageSize) || 1
  const activeModelMeta = availableModels.find(m => m.id === selectedModelId) || availableModels[0]
  const championModel = availableModels.find(m => m.isChampion) || availableModels[0]

  return (
    <div className="container" style={{ padding: '3rem 0 6rem' }}>
      {/* ── Top Header & Controls ── */}
      <div style={{ marginBottom: '2rem', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1.5rem' }}>
        <div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <span className="clinical-badge" style={{ background: '#F0FDFA', color: '#0D9488', borderColor: '#99F6E4' }}>
              <CheckCircle2 size={13} /> Empirical Benchmark Evaluation
            </span>
            <span className="clinical-badge navy">
              Held-Out Test Split (20% Stratified)
            </span>
          </div>

          <h1 style={{ fontSize: '2.4rem', fontWeight: 800, color: '#0A192F', letterSpacing: '-0.02em', margin: 0 }}>
            Model Evaluation Dashboard
          </h1>
          <p style={{ color: '#64748B', fontSize: '1.05rem', marginTop: '0.35rem', maxWidth: 740 }}>
            Inspect and compare classification performance, confusion matrices, and error modes across architectures.
          </p>
        </div>

        {/* Action Controls & Export Buttons */}
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <a
            href={`/api/evaluation/export/csv?model=${encodeURIComponent(selectedModelId)}`}
            className="btn-secondary"
            download={`dentura_ai_${selectedModelId}_evaluation.csv`}
            style={{ fontSize: '0.85rem', padding: '0.65rem 1.1rem' }}
          >
            <FileSpreadsheet size={16} /> Export CSV
          </a>
          <a
            href={`/api/evaluation/export/json?model=${encodeURIComponent(selectedModelId)}`}
            className="btn-secondary"
            download={`dentura_ai_${selectedModelId}_evaluation_report.json`}
            style={{ fontSize: '0.85rem', padding: '0.65rem 1.1rem' }}
          >
            <FileCode size={16} /> Export JSON
          </a>
          {selectedModelId === 'efficientnet_b0' && (
            <button
              className="btn-primary"
              onClick={handleRecalculate}
              disabled={isRefreshing}
              style={{ fontSize: '0.85rem', padding: '0.65rem 1.2rem' }}
            >
              <RefreshCw size={16} className={isRefreshing ? 'spinner' : ''} />
              {isRefreshing ? 'Recalculating...' : 'Recalculate'}
            </button>
          )}
        </div>
      </div>

      {/* ── Interactive Model Switcher Cards ── */}
      <div style={{ marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
          <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0A192F', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <Layers size={18} color="#0D9488" /> Select Evaluation Architecture ({availableModels.length} Models Tested)
          </div>
          <span style={{ fontSize: '0.8rem', color: '#64748B' }}>
            Click any architecture to switch metrics & confusion matrices
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem' }}>
          {availableModels.map(model => {
            const isSelected = model.id === selectedModelId
            const accDiff = championModel && model.id !== championModel.id
              ? (model.accuracy - championModel.accuracy) * 100
              : 0

            return (
              <div
                key={model.id}
                onClick={() => handleModelChange(model.id)}
                style={{
                  background: isSelected ? '#F0FDFA' : '#FFFFFF',
                  border: isSelected ? '2px solid #0D9488' : '1px solid #E2E8F0',
                  borderRadius: 16,
                  padding: '1.4rem',
                  cursor: 'pointer',
                  transition: 'all 180ms ease',
                  boxShadow: isSelected ? '0 8px 24px -4px rgba(13,148,136,0.18)' : '0 2px 6px rgba(10,25,47,0.04)',
                  position: 'relative'
                }}
              >
                {/* Header Tag */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                  <span
                    className={`clinical-badge ${model.isChampion ? '' : 'navy'}`}
                    style={{ fontSize: '0.7rem' }}
                  >
                    {model.badge}
                  </span>
                  {model.isChampion ? (
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#0D9488', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                      <Sparkles size={13} /> Deployed
                    </span>
                  ) : (
                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#EF4444' }}>
                      {accDiff.toFixed(1)}% vs Champion
                    </span>
                  )}
                </div>

                {/* Name */}
                <h4 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0A192F', margin: '0 0 0.35rem 0' }}>
                  {model.name}
                </h4>
                <p style={{ fontSize: '0.82rem', color: '#64748B', margin: '0 0 1rem 0', minHeight: 38 }}>
                  {model.description}
                </p>

                {/* Key Metrics Row */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid #E2E8F0', paddingTop: '0.85rem' }}>
                  <div>
                    <div style={{ fontSize: '0.7rem', color: '#64748B', textTransform: 'uppercase', fontWeight: 600 }}>Accuracy</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 800, color: isSelected ? '#0D9488' : '#0A192F' }}>
                      {(model.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.7rem', color: '#64748B', textTransform: 'uppercase', fontWeight: 600 }}>F1-Score</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 800, color: isSelected ? '#0D9488' : '#0A192F' }}>
                      {(model.f1Score * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.7rem', color: '#64748B', textTransform: 'uppercase', fontWeight: 600 }}>Params</div>
                    <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#475569', marginTop: 4 }}>
                      {model.parameters.split(' ')[0]}
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Active Architecture Details Bar ── */}
      <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 14, padding: '1rem 1.5rem', marginBottom: '2.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          <div style={{ width: 38, height: 38, borderRadius: 10, background: '#F0FDFA', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0D9488' }}>
            <Cpu size={20} />
          </div>
          <div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0A192F' }}>
              Active Evaluation: {metadata.modelName}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748B' }}>
              Architecture: {metadata.modelType || activeModelMeta?.type} • Checkpoint: <code>{metadata.checkpointPath}</code>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className="clinical-badge" style={{ background: '#F8FAFC', color: '#334155' }}>
            {metadata.zeroDivisionHandling}
          </span>
          <span className="clinical-badge" style={{ background: '#F0FDFA', color: '#0D9488' }}>
            {metadata.classOrder.length} Target Classes
          </span>
        </div>
      </div>

      {/* ── Overview Stat Cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem', marginBottom: '2.5rem' }}>
        <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 16, padding: '1.5rem', boxShadow: '0 2px 6px rgba(10,25,47,0.04)' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Test Accuracy
          </div>
          <div style={{ fontSize: '2.4rem', fontWeight: 800, color: '#0D9488', margin: '0.4rem 0 0.2rem' }}>
            {(overallMetrics.accuracy * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: '0.82rem', color: '#475569' }}>
            <strong>{overallMetrics.correctPredictions}</strong> of {overallMetrics.totalSamples} correct
          </div>
        </div>

        <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 16, padding: '1.5rem', boxShadow: '0 2px 6px rgba(10,25,47,0.04)' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Macro F1-Score
          </div>
          <div style={{ fontSize: '2.4rem', fontWeight: 800, color: '#0A192F', margin: '0.4rem 0 0.2rem' }}>
            {(overallMetrics.macroF1 * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: '0.82rem', color: '#475569' }}>
            Unweighted mean across all 6 classes
          </div>
        </div>

        <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 16, padding: '1.5rem', boxShadow: '0 2px 6px rgba(10,25,47,0.04)' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Balanced Accuracy
          </div>
          <div style={{ fontSize: '2.4rem', fontWeight: 800, color: '#0A192F', margin: '0.4rem 0 0.2rem' }}>
            {(overallMetrics.balancedAccuracy * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: '0.82rem', color: '#475569' }}>
            Normalized for class prevalence
          </div>
        </div>

        <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 16, padding: '1.5rem', boxShadow: '0 2px 6px rgba(10,25,47,0.04)' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Evaluated Samples
          </div>
          <div style={{ fontSize: '2.4rem', fontWeight: 800, color: '#0A192F', margin: '0.4rem 0 0.2rem' }}>
            {overallMetrics.totalSamples.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.82rem', color: '#EF4444', fontWeight: 600 }}>
            {overallMetrics.incorrectPredictions} misclassifications ({((overallMetrics.incorrectPredictions / overallMetrics.totalSamples) * 100).toFixed(1)}%)
          </div>
        </div>
      </div>

      {/* ── Scientific Integrity & Data Leakage Audit Alert ── */}
      {integrityAudit && integrityAudit.hasLeakage && (
        <div
          style={{
            background: '#FFFBEB',
            border: '1px solid #FDE68A',
            borderRadius: 16,
            padding: '1.5rem 1.75rem',
            marginBottom: '2.5rem',
            boxShadow: '0 2px 8px rgba(245,158,11,0.08)'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
            <ShieldAlert size={26} color="#D97706" style={{ flexShrink: 0, marginTop: 2 }} />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#92400E', margin: 0 }}>
                  Scientific Integrity Alert: Source Dataset Leakage & Ambiguity Detected
                </h4>
                <span className="clinical-badge warning" style={{ fontSize: '0.7rem' }}>
                  Audit Finding: {integrityAudit.duplicateCount} Duplicates ({integrityAudit.leakageRate}% of Test Set)
                </span>
              </div>
              <p style={{ fontSize: '0.88rem', color: '#B45309', margin: '0.5rem 0 0.75rem', lineHeight: 1.55 }}>
                {integrityAudit.summary}
              </p>
              <div style={{ fontSize: '0.82rem', color: '#92400E', background: 'rgba(254, 243, 199, 0.6)', padding: '0.6rem 0.85rem', borderRadius: 8 }}>
                <strong>Key Pathology Confusion Driver:</strong> The raw Kaggle dataset contains identical images filed under both <code>Calculus</code> and <code>Gingivitis</code>. This explains why the largest confusion mass occurs between these two classes across all 3 evaluated architectures.
                {scientificDisclaimer && (
                  <div style={{ marginTop: '0.4rem', borderTop: '1px dashed #FDE68A', paddingTop: '0.4rem' }}>
                    <strong>Scientific Protocol Note:</strong> {scientificDisclaimer}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Confusion Matrix Section ── */}
      <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 20, padding: '2rem', marginBottom: '2.5rem', boxShadow: '0 4px 12px rgba(10,25,47,0.04)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#0A192F', margin: 0 }}>
              Confusion Matrix — {metadata.modelName}
            </h3>
            <p style={{ color: '#64748B', fontSize: '0.88rem', margin: '0.2rem 0 0' }}>
              Rows represent ground truth classes; columns represent model predictions.
            </p>
          </div>

          {/* Toggle Raw vs Normalized */}
          <div style={{ display: 'flex', background: '#F1F5F9', padding: '0.25rem', borderRadius: 9999 }}>
            <button
              onClick={() => setMatrixView('raw')}
              style={{
                padding: '0.45rem 1rem',
                fontSize: '0.82rem',
                fontWeight: 600,
                borderRadius: 9999,
                background: matrixView === 'raw' ? '#FFFFFF' : 'transparent',
                color: matrixView === 'raw' ? '#0D9488' : '#64748B',
                boxShadow: matrixView === 'raw' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
              }}
            >
              Raw Counts
            </button>
            <button
              onClick={() => setMatrixView('normalized')}
              style={{
                padding: '0.45rem 1rem',
                fontSize: '0.82rem',
                fontWeight: 600,
                borderRadius: 9999,
                background: matrixView === 'normalized' ? '#FFFFFF' : 'transparent',
                color: matrixView === 'normalized' ? '#0D9488' : '#64748B',
                boxShadow: matrixView === 'normalized' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
              }}
            >
              Row-Normalized (Recall %)
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem', alignItems: 'start' }}>
          {/* Matrix Table */}
          <div style={{ overflowX: 'auto' }}>
            <div style={{ textAlign: 'center', fontSize: '0.78rem', fontWeight: 700, color: '#64748B', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
              Predicted Class →
            </div>

            <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 4 }}>
              <thead>
                <tr>
                  <th style={{ padding: '0.5rem', fontSize: '0.75rem', color: '#64748B', textAlign: 'right' }}>
                    True Class ↓
                  </th>
                  {classLabels.map(cls => (
                    <th key={cls} style={{ padding: '0.5rem 0.35rem', fontSize: '0.78rem', fontWeight: 700, color: '#0A192F', textAlign: 'center', minWidth: 70 }}>
                      {cls.replace('_', ' ')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {classLabels.map((trueCls, rIdx) => (
                  <tr key={trueCls}>
                    <td style={{ padding: '0.5rem 0.75rem', fontSize: '0.82rem', fontWeight: 700, color: '#0A192F', textAlign: 'right', whiteSpace: 'nowrap' }}>
                      {trueCls.replace('_', ' ')}
                    </td>
                    {classLabels.map((predCls, cIdx) => {
                      const isDiag = rIdx === cIdx
                      const rawVal = rawMatrix[rIdx]?.[cIdx] ?? 0
                      const normVal = normalizedMatrix[rIdx]?.[cIdx] ?? 0
                      const isSelected = selectedCell?.row === rIdx && selectedCell?.col === cIdx

                      let bg = '#F8FAFC'
                      let textColor = '#475569'

                      if (isDiag) {
                        const intensity = normVal
                        bg = `rgba(13, 148, 136, ${Math.max(0.12, intensity * 0.95)})`
                        textColor = intensity > 0.5 ? '#FFFFFF' : '#0F766E'
                      } else if (rawVal > 0) {
                        bg = `rgba(239, 68, 68, ${Math.min(0.7, 0.08 + normVal * 1.5)})`
                        textColor = normVal > 0.15 ? '#FFFFFF' : '#991B1B'
                      }

                      return (
                        <td
                          key={predCls}
                          onClick={() => setSelectedCell({ row: rIdx, col: cIdx })}
                          onMouseEnter={() => setSelectedCell({ row: rIdx, col: cIdx })}
                          style={{
                            background: bg,
                            color: textColor,
                            padding: '0.85rem 0.5rem',
                            textAlign: 'center',
                            borderRadius: 8,
                            fontSize: '0.88rem',
                            fontWeight: isDiag ? 800 : rawVal > 0 ? 600 : 400,
                            cursor: 'pointer',
                            outline: isSelected ? '2px solid #0D9488' : 'none',
                            transition: 'all 120ms ease'
                          }}
                        >
                          {matrixView === 'raw' ? rawVal : `${(normVal * 100).toFixed(1)}%`}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Cell inspection tooltip banner */}
            {selectedCell && (
              <div style={{ marginTop: '1rem', background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: 8, padding: '0.65rem 1rem', fontSize: '0.85rem', color: '#334155', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <strong>True:</strong> {classLabels[selectedCell.row]?.replace('_', ' ')} →{' '}
                  <strong>Predicted:</strong> {classLabels[selectedCell.col]?.replace('_', ' ')}
                </div>
                <div>
                  Count: <strong>{rawMatrix[selectedCell.row]?.[selectedCell.col]}</strong> (
                  {((normalizedMatrix[selectedCell.row]?.[selectedCell.col] || 0) * 100).toFixed(1)}% of row)
                </div>
              </div>
            )}
          </div>

          {/* Top Confused Pairs Panel */}
          <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: 14, padding: '1.25rem' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0A192F', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <AlertTriangle size={16} color="#F59E0B" /> Top Confused Class Pairs
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
              {topConfusedPairs.slice(0, 6).map((pair, idx) => (
                <div
                  key={idx}
                  style={{
                    background: '#FFFFFF',
                    border: '1px solid #E2E8F0',
                    borderRadius: 8,
                    padding: '0.6rem 0.8rem',
                    fontSize: '0.82rem'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600, color: '#1E293B' }}>
                    <span>
                      {pair.trueClass.replace('_', ' ')} → {pair.predictedClass.replace('_', ' ')}
                    </span>
                    <span style={{ color: '#EF4444', fontWeight: 700 }}>
                      {pair.count} errors ({pair.percentageOfTrue}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Per-Class Performance Table & Metrics ── */}
      <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 20, padding: '2rem', marginBottom: '2.5rem', boxShadow: '0 4px 12px rgba(10,25,47,0.04)' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <h3 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#0A192F', margin: 0 }}>
            Per-Class Diagnostic Performance — {metadata.modelName}
          </h3>
          <p style={{ color: '#64748B', fontSize: '0.88rem', margin: '0.2rem 0 0' }}>
            Click column headers to sort metrics across classes.
          </p>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#F8FAFC', borderBottom: '1px solid #E2E8F0' }}>
                <th
                  onClick={() => handleSort('className')}
                  style={{ padding: '0.85rem 1rem', fontSize: '0.8rem', fontWeight: 700, color: '#475569', cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.04em' }}
                >
                  Class Name {sortColumn === 'className' && (sortAsc ? '▲' : '▼')}
                </th>
                <th
                  onClick={() => handleSort('support')}
                  style={{ padding: '0.85rem 1rem', fontSize: '0.8rem', fontWeight: 700, color: '#475569', cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.04em' }}
                >
                  Support {sortColumn === 'support' && (sortAsc ? '▲' : '▼')}
                </th>
                <th
                  onClick={() => handleSort('truePositives')}
                  style={{ padding: '0.85rem 1rem', fontSize: '0.8rem', fontWeight: 700, color: '#475569', cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.04em' }}
                >
                  TP / FP / FN
                </th>
                <th
                  onClick={() => handleSort('precision')}
                  style={{ padding: '0.85rem 1rem', fontSize: '0.8rem', fontWeight: 700, color: '#475569', cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.04em' }}
                >
                  Precision {sortColumn === 'precision' && (sortAsc ? '▲' : '▼')}
                </th>
                <th
                  onClick={() => handleSort('recall')}
                  style={{ padding: '0.85rem 1rem', fontSize: '0.8rem', fontWeight: 700, color: '#475569', cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.04em' }}
                >
                  Recall {sortColumn === 'recall' && (sortAsc ? '▲' : '▼')}
                </th>
                <th
                  onClick={() => handleSort('f1Score')}
                  style={{ padding: '0.85rem 1rem', fontSize: '0.8rem', fontWeight: 700, color: '#475569', cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.04em' }}
                >
                  F1-Score {sortColumn === 'f1Score' && (sortAsc ? '▲' : '▼')}
                </th>
                <th style={{ padding: '0.85rem 1rem', fontSize: '0.8rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.04em', width: 220 }}>
                  Visual Balance
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedClassMetrics.map(item => (
                <tr key={item.className} style={{ borderBottom: '1px solid #F1F5F9' }}>
                  <td style={{ padding: '1rem', fontWeight: 700, color: '#0A192F', fontSize: '0.92rem' }}>
                    {item.displayName}
                  </td>
                  <td style={{ padding: '1rem', color: '#475569', fontSize: '0.9rem' }}>
                    {item.support}
                  </td>
                  <td style={{ padding: '1rem', fontSize: '0.85rem', color: '#64748B' }}>
                    <span style={{ color: '#0D9488', fontWeight: 700 }}>{item.truePositives}</span> /{' '}
                    <span style={{ color: '#EF4444' }}>{item.falsePositives}</span> /{' '}
                    <span style={{ color: '#F59E0B' }}>{item.falseNegatives}</span>
                  </td>
                  <td style={{ padding: '1rem', fontWeight: 600, color: '#1E293B', fontSize: '0.9rem' }}>
                    {(item.precision * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '1rem', fontWeight: 600, color: '#1E293B', fontSize: '0.9rem' }}>
                    {(item.recall * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '1rem', fontWeight: 800, color: '#0D9488', fontSize: '0.95rem' }}>
                    {(item.f1Score * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.72rem' }}>
                        <span style={{ width: 24, color: '#64748B' }}>F1</span>
                        <div style={{ flex: 1, height: 8, background: '#F1F5F9', borderRadius: 4, overflow: 'hidden' }}>
                          <div style={{ width: `${item.f1Score * 100}%`, height: '100%', background: '#0D9488', borderRadius: 4 }} />
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.72rem' }}>
                        <span style={{ width: 24, color: '#64748B' }}>Rec</span>
                        <div style={{ flex: 1, height: 8, background: '#F1F5F9', borderRadius: 4, overflow: 'hidden' }}>
                          <div style={{ width: `${item.recall * 100}%`, height: '100%', background: '#06B6D4', borderRadius: 4 }} />
                        </div>
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Error Analysis & Misclassified Scans (for Champion model with live weights) ── */}
      {selectedModelId === 'efficientnet_b0' ? (
        <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 20, padding: '2rem', boxShadow: '0 4px 12px rgba(10,25,47,0.04)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#0A192F', margin: 0 }}>
                Error Analysis & Misclassified Scans — {metadata.modelName}
              </h3>
              <p style={{ color: '#64748B', fontSize: '0.88rem', margin: '0.2rem 0 0' }}>
                Inspect individual failure modes across test samples without exposing absolute patient or server paths.
              </p>
            </div>

            {/* Filters */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#475569' }}>True Class:</span>
                <select
                  value={trueClassFilter}
                  onChange={e => {
                    setTrueClassFilter(e.target.value)
                    setPage(1)
                  }}
                  style={{ padding: '0.4rem 0.75rem', borderRadius: 8, border: '1px solid #CBD5E1', fontSize: '0.85rem', background: '#FFFFFF', color: '#1E293B' }}
                >
                  <option value="all">All Classes</option>
                  {classLabels.map(cls => (
                    <option key={cls} value={cls}>
                      {cls.replace('_', ' ')}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#475569' }}>Predicted Class:</span>
                <select
                  value={predClassFilter}
                  onChange={e => {
                    setPredClassFilter(e.target.value)
                    setPage(1)
                  }}
                  style={{ padding: '0.4rem 0.75rem', borderRadius: 8, border: '1px solid #CBD5E1', fontSize: '0.85rem', background: '#FFFFFF', color: '#1E293B' }}
                >
                  <option value="all">All Classes</option>
                  {classLabels.map(cls => (
                    <option key={cls} value={cls}>
                      {cls.replace('_', ' ')}
                    </option>
                  ))}
                </select>
              </div>

              <span className="clinical-badge" style={{ fontSize: '0.72rem' }}>
                Showing {totalMisclassified} misclassified cases
              </span>
            </div>
          </div>

          {/* Misclassified Cards Grid */}
          {misclassifiedSamples.length === 0 ? (
            <div style={{ padding: '3rem', textAlign: 'center', color: '#64748B' }}>
              <CheckCircle2 size={36} color="#10B981" style={{ margin: '0 auto 0.75rem' }} />
              <p>No misclassified samples match the selected filter criteria.</p>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1.25rem', marginBottom: '1.75rem' }}>
              {misclassifiedSamples.map(sample => (
                <div
                  key={sample.id}
                  style={{
                    background: '#F8FAFC',
                    border: '1px solid #E2E8F0',
                    borderRadius: 12,
                    overflow: 'hidden',
                    display: 'flex',
                    flexDirection: 'column'
                  }}
                >
                  {/* Image */}
                  <div style={{ width: '100%', height: 180, background: '#000', position: 'relative' }}>
                    <img
                      src={`/api/evaluation/image/${encodeURIComponent(sample.trueClass)}/${encodeURIComponent(sample.filename)}`}
                      alt={`Misclassified ${sample.id}`}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      loading="lazy"
                    />
                    <div
                      style={{
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        background: 'rgba(10,25,47,0.85)',
                        backdropFilter: 'blur(4px)',
                        color: '#FFF',
                        fontSize: '0.7rem',
                        padding: '0.2rem 0.5rem',
                        borderRadius: 4
                      }}
                    >
                      ID: {sample.filename}
                    </div>
                  </div>

                  {/* Details */}
                  <div style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                      <span style={{ color: '#475569' }}>Ground Truth:</span>
                      <span style={{ fontWeight: 700, color: '#0F766E', background: '#CCFBF1', padding: '0.15rem 0.5rem', borderRadius: 4 }}>
                        {sample.trueClass.replace('_', ' ')}
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                      <span style={{ color: '#475569' }}>Predicted:</span>
                      <span style={{ fontWeight: 700, color: '#991B1B', background: '#FEE2E2', padding: '0.15rem 0.5rem', borderRadius: 4 }}>
                        {sample.predictedClass.replace('_', ' ')} ({(sample.predictedConfidence * 100).toFixed(1)}%)
                      </span>
                    </div>

                    {/* Top probabilities breakdown */}
                    <div style={{ marginTop: '0.5rem', borderTop: '1px solid #E2E8F0', paddingTop: '0.5rem' }}>
                      <div style={{ fontSize: '0.72rem', color: '#64748B', fontWeight: 600, marginBottom: 4 }}>
                        Model Output Distribution:
                      </div>
                      {(Object.entries(sample.probabilities) as [string, number][])
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 3)
                        .map(([cls, prob]) => (
                          <div key={cls} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#334155' }}>
                            <span>{cls.replace('_', ' ')}</span>
                            <span style={{ fontWeight: 600 }}>{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Pagination Bar */}
          {totalPages > 1 && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid #E2E8F0', paddingTop: '1.25rem' }}>
              <span style={{ fontSize: '0.85rem', color: '#64748B' }}>
                Page <strong>{page}</strong> of <strong>{totalPages}</strong> ({totalMisclassified} misclassified total)
              </span>

              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  className="btn-secondary"
                  disabled={page <= 1}
                  onClick={() => setPage(page - 1)}
                  style={{ padding: '0.4rem 0.8rem', fontSize: '0.82rem' }}
                >
                  <ChevronLeft size={16} /> Previous
                </button>
                <button
                  className="btn-secondary"
                  disabled={page >= totalPages}
                  onClick={() => setPage(page + 1)}
                  style={{ padding: '0.4rem 0.8rem', fontSize: '0.82rem' }}
                >
                  Next <ChevronRight size={16} />
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Notice for comparative models */
        <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 20, padding: '2.5rem', textAlign: 'center', boxShadow: '0 4px 12px rgba(10,25,47,0.04)' }}>
          <Zap size={36} color="#0D9488" style={{ margin: '0 auto 0.75rem' }} />
          <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0A192F', margin: '0 0 0.5rem 0' }}>
            Comparative Research Benchmark
          </h3>
          <p style={{ color: '#64748B', maxWidth: 560, margin: '0 auto 1.25rem', fontSize: '0.9rem', lineHeight: 1.55 }}>
            {metadata.modelName} was evaluated during the author's comparative study as a baseline architecture.
            The author selected <strong>EfficientNetB0 (82.51% Accuracy)</strong> as the sole production checkpoint.
            Switch back to EfficientNetB0 to inspect individual sample image previews and Grad-CAM activations.
          </p>
          <button
            className="btn-primary"
            onClick={() => handleModelChange('efficientnet_b0')}
            style={{ fontSize: '0.88rem' }}
          >
            Switch to Deployed EfficientNetB0
          </button>
        </div>
      )}
    </div>
  )
}
