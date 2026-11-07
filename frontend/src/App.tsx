import React, { useState, useEffect, useRef } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Eye,
  Microscope,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  UploadCloud,
  Stethoscope
} from 'lucide-react'
import './App.css'
import { EvaluationDashboard } from './components/EvaluationDashboard'

interface ProbabilityItem {
  classKey: string
  displayName: string
  icon: string
  probability: number
  badgeColor: string
  severity: string
}

interface DiseaseInfo {
  displayName: string
  icon: string
  badgeColor: string
  accentColor: string
  severity: string
  description: string
  clinicalNotes: string
  urgency: string
}

interface PredictionResult {
  predictedClass: string
  displayName: string
  icon: string
  confidence: number
  isUncertain: boolean
  threshold: number
  warningMessage: string | null
  diseaseInfo: DiseaseInfo
  probabilities: ProbabilityItem[]
}

interface GradcamResult {
  supported: boolean
  targetLayer: string
  predictedClass: string
  confidence: number
  originalBase64: string
  heatmapBase64: string
  overlayBase64: string
}

interface SampleImage {
  id: string
  filename: string
  label: string
  description: string
  thumbnail: string
}

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<'landing' | 'studio' | 'evaluation'>('landing')
  const [activeModel, setActiveModel] = useState<'efficientnet_b0' | 'custom_cnn' | 'resnet_50'>('efficientnet_b0')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [samples, setSamples] = useState<SampleImage[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [gradcam, setGradcam] = useState<GradcamResult | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [gradcamViewMode, setGradcamViewMode] = useState<'all' | 'overlay' | 'heatmap' | 'original'>('all')
  const [isDragOver, setIsDragOver] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // Fetch sample images from backend on mount
  useEffect(() => {
    fetch('/api/samples')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setSamples(data)
      })
      .catch(err => console.log('Notice: Could not load samples yet:', err))
  }, [])

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please upload a valid image file (JPEG, PNG, or WebP).')
      return
    }
    setErrorMsg(null)
    setSelectedFile(file)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setPrediction(null)
    setGradcam(null)
  }

  const handleSampleClick = async (sample: SampleImage) => {
    try {
      setIsAnalyzing(true)
      setErrorMsg(null)
      setCurrentView('studio')

      // Fetch raw sample file from backend
      const res = await fetch(`/api/samples/${sample.id}/raw`)
      if (!res.ok) throw new Error('Failed to load sample image file')
      const blob = await res.blob()
      const file = new File([blob], sample.filename, { type: 'image/jpeg' })

      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(blob))

      // Trigger full analysis
      const formData = new FormData()
      formData.append('file', file)

      const analyzeRes = await fetch('/api/analyze-full', {
        method: 'POST',
        body: formData
      })
      if (!analyzeRes.ok) throw new Error('Analysis request failed')

      const data = await analyzeRes.json()
      setPrediction(data.prediction)
      setGradcam(data.gradcam)
    } catch (err: any) {
      setErrorMsg(err.message || 'Error loading sample image.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setIsAnalyzing(true)
    setErrorMsg(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const res = await fetch('/api/analyze-full', {
        method: 'POST',
        body: formData
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Inference error occurred.')
      }

      const data = await res.json()
      setPrediction(data.prediction)
      setGradcam(data.gradcam)
    } catch (err: any) {
      setErrorMsg(err.message || 'Error analyzing oral image.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setPrediction(null)
    setGradcam(null)
    setErrorMsg(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="app-wrapper">
      {/* ── Top Navigation Bar ── */}
      <nav className="navbar">
        <div className="container nav-container">
          <div className="nav-brand" onClick={() => setCurrentView('landing')} style={{ cursor: 'pointer' }}>
            <div className="nav-brand-icon">
              <Stethoscope size={22} />
            </div>
            <span>Dentura AI</span>
            <span className="nav-brand-badge">Alternative</span>
          </div>

          <div className="nav-links">
            <button
              className={`nav-link ${currentView === 'landing' ? 'active' : ''}`}
              onClick={() => setCurrentView('landing')}
            >
              Overview
            </button>
            <button
              className={`nav-link ${currentView === 'evaluation' ? 'active' : ''}`}
              onClick={() => setCurrentView('evaluation')}
            >
              Model Evaluation
            </button>
            <button
              className={`nav-link ${currentView === 'studio' ? 'active' : ''}`}
              onClick={() => setCurrentView('studio')}
            >
              Analysis Studio
            </button>
            <a href="#benchmarks" className="nav-link" onClick={() => setCurrentView('landing')}>
              Benchmarks
            </a>
            <a href="#diseases" className="nav-link" onClick={() => setCurrentView('landing')}>
              Disease Index
            </a>
          </div>

          <div className="nav-actions">
            <div
              className="clinical-badge"
              style={{ cursor: 'pointer' }}
              title="Click to view Model Evaluation Dashboard"
              onClick={() => setCurrentView('evaluation')}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: activeModel === 'efficientnet_b0' ? '#10B981' : '#F59E0B',
                  display: 'inline-block'
                }}
              />
              {activeModel === 'efficientnet_b0'
                ? 'EffNet-B0 (82.5%)'
                : activeModel === 'custom_cnn'
                ? 'Custom CNN (75.8%)'
                : 'ResNet-50 (73.2%)'}
            </div>
            <button
              className="btn-primary nav-cta-btn"
              onClick={() => setCurrentView(currentView === 'studio' ? 'landing' : 'studio')}
            >
              {currentView === 'studio' ? (
                'Overview'
              ) : (
                <>
                  Launch Studio <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>
        </div>
      </nav>

      {/* ── Main View Router ── */}
      {currentView === 'landing' ? (
        <main>
          {/* ── Hero Section ── */}
          <section className="hero-section">
            <div className="container hero-content">
              <div className="hero-badge-wrap">
                <span className="clinical-badge">
                  <Sparkles size={14} /> AI-Assisted Oral Pathology Classification
                </span>
              </div>

              <h1 className="hero-title">
                Precision Dental AI with <br />
                <span className="teal-gradient-text">Pixel-Level Spatial Explainability</span>
              </h1>

              <p className="hero-subtitle">
                Deploying fine-tuned EfficientNetB0 deep learning to classify six distinct oral and periodontal conditions.
                Integrated with real-time Grad-CAM gradient backpropagation for verifiable clinical interpretability.
              </p>

              <div className="hero-cta-group">
                <button
                  className="btn-primary"
                  style={{ padding: '0.9rem 2rem', fontSize: '1.05rem' }}
                  onClick={() => setCurrentView('studio')}
                >
                  Start Clinical Analysis <ArrowRight size={18} />
                </button>
                <a href="#benchmarks" className="btn-secondary" style={{ padding: '0.9rem 1.8rem', fontSize: '1.05rem' }}>
                  Explore Benchmarks
                </a>
              </div>

              {/* Stats Row */}
              <div className="hero-stats-row">
                <div className="stat-item">
                  <div className="stat-number">82.5%</div>
                  <div className="stat-label">Classification F1-Score</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number">6</div>
                  <div className="stat-label">Detected Pathologies</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number">&lt; 0.4s</div>
                  <div className="stat-label">Inference Latency</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number">Grad-CAM</div>
                  <div className="stat-label">Spatial Heatmap Localization</div>
                </div>
              </div>
            </div>
          </section>

          {/* ── Interactive Product Preview ── */}
          <section className="preview-section">
            <div className="container">
              <div className="section-header">
                <span className="section-tag">Clinical Studio Preview</span>
                <h2 className="section-title">Automated Lesion Analysis & Localization</h2>
                <p className="section-desc">
                  Explore how Dentura AI translates standard photographic oral captures into diagnostic probability vectors and spatial lesion heatmaps.
                </p>
              </div>

              <div className="preview-card-showcase">
                <div className="preview-image-side">
                  <img
                    src="https://images.unsplash.com/photo-1606811841689-23dfddce3e95?auto=format&fit=crop&w=800&q=80"
                    alt="Oral clinical examination preview"
                  />
                  <div className="preview-badge-overlay">
                    <Activity size={14} color="#2DD4BF" />
                    <span>Grad-CAM Activation: top_activation</span>
                  </div>
                </div>

                <div className="preview-data-side">
                  <div className="preview-confidence-panel">
                    <div className="preview-diag-header">
                      <div className="preview-diag-title">
                        <span>🦷 Dental Caries (Cavity)</span>
                      </div>
                      <div className="preview-score">99.3%</div>
                    </div>
                    <p style={{ fontSize: '0.88rem', color: '#0F766E', margin: 0 }}>
                      Primary pathological classification with optimal confidence threshold.
                    </p>
                  </div>

                  <div className="preview-bars-preview">
                    <div className="mini-prob-row">
                      <span className="mini-prob-name">🦷 Caries</span>
                      <div className="mini-bar-track">
                        <div className="mini-bar-fill" style={{ width: '99.3%', background: '#EF4444' }} />
                      </div>
                      <span className="mini-prob-pct">99.3%</span>
                    </div>

                    <div className="mini-prob-row">
                      <span className="mini-prob-name">🪨 Calculus</span>
                      <div className="mini-bar-track">
                        <div className="mini-bar-fill" style={{ width: '0.5%', background: '#F59E0B' }} />
                      </div>
                      <span className="mini-prob-pct">0.5%</span>
                    </div>

                    <div className="mini-prob-row">
                      <span className="mini-prob-name">🩸 Gingivitis</span>
                      <div className="mini-bar-track">
                        <div className="mini-bar-fill" style={{ width: '0.1%', background: '#EC4899' }} />
                      </div>
                      <span className="mini-prob-pct">0.1%</span>
                    </div>

                    <div className="mini-prob-row">
                      <span className="mini-prob-name">🎨 Discoloration</span>
                      <div className="mini-bar-track">
                        <div className="mini-bar-fill" style={{ width: '0.1%', background: '#0D9488' }} />
                      </div>
                      <span className="mini-prob-pct">0.1%</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
                    <button className="btn-primary" onClick={() => setCurrentView('studio')} style={{ flex: 1 }}>
                      Try with Live Scan <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ── How It Works Section ── */}
          <section id="how-it-works" className="how-section">
            <div className="container">
              <div className="section-header">
                <span className="section-tag">Diagnostic Pipeline</span>
                <h2 className="section-title">How Dentura AI Analyzes Oral Scans</h2>
                <p className="section-desc">
                  An end-to-end clinical computer vision workflow engineered for transparency and accuracy.
                </p>
              </div>

              <div className="steps-grid">
                <div className="step-card">
                  <div className="step-number">1</div>
                  <h3 className="step-title">Tensor Standardization</h3>
                  <p className="step-text">
                    Uploaded clinical photos are converted to high-definition RGB tensors, resized to 224×224 pixels, and normalized using EfficientNet-specific zero-centering.
                  </p>
                </div>

                <div className="step-card">
                  <div className="step-number">2</div>
                  <h3 className="step-title">Deep Feature Extraction</h3>
                  <p className="step-text">
                    The fine-tuned EfficientNetB0 backbone processes depthwise separable convolutions to identify microstructural features of enamel demineralization, calculus accretion, and mucosal erythema.
                  </p>
                </div>

                <div className="step-card">
                  <div className="step-number">3</div>
                  <h3 className="step-title">Grad-CAM Heatmap Generation</h3>
                  <p className="step-text">
                    Gradients of the predicted disease score are backpropagated to the final convolutional feature layer (top_activation, 7×7×1280), generating an authentic spatial activation heatmap overlay.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* ── Clinical Performance Benchmarks ── */}
          <section id="benchmarks" className="benchmark-section">
            <div className="container">
              <div className="section-header">
                <span className="section-tag">Empirical Results</span>
                <h2 className="section-title">Architectural Comparative Analysis</h2>
                <p className="section-desc">
                  Results from the comparative training study on the stratified oral disease benchmark dataset.
                </p>
              </div>

              <div className="benchmark-table-card">
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Model Architecture</th>
                      <th>Model Type</th>
                      <th>Param Count</th>
                      <th>Accuracy</th>
                      <th>F1-Score</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="highlighted">
                      <td>
                        <div className="model-name-cell">
                          <CheckCircle2 size={18} color="#0D9488" /> EfficientNetB0 (Fine-Tuned)
                        </div>
                      </td>
                      <td>Transfer Learning</td>
                      <td>~4.06M</td>
                      <td>82.51%</td>
                      <td>82.51%</td>
                      <td>
                        <span className="clinical-badge">Deployed</span>
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <div className="model-name-cell">Custom 4-Block CNN</div>
                      </td>
                      <td>Scratch Conv2D + GAP</td>
                      <td>~457K</td>
                      <td>75.78%</td>
                      <td>75.18%</td>
                      <td>Evaluated</td>
                    </tr>
                    <tr>
                      <td>
                        <div className="model-name-cell">ResNet50 Backbone</div>
                      </td>
                      <td>Transfer Learning</td>
                      <td>~23.6M</td>
                      <td>73.18%</td>
                      <td>72.59%</td>
                      <td>Evaluated</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div style={{ textAlign: 'center', marginTop: '1.75rem' }}>
                <button
                  className="btn-primary"
                  onClick={() => setCurrentView('evaluation')}
                  style={{ padding: '0.8rem 1.8rem' }}
                >
                  <Activity size={18} /> Open Interactive Model Evaluation Dashboard
                </button>
              </div>
            </div>
          </section>

          {/* ── 6 Oral Pathologies Index ── */}
          <section id="diseases" className="diseases-section">
            <div className="container">
              <div className="section-header">
                <span className="section-tag">Disease Taxonomy</span>
                <h2 className="section-title">Classifiable Oral Pathologies</h2>
                <p className="section-desc">
                  The model has been specifically trained to classify oral conditions into six categorical conditions.
                </p>
              </div>

              <div className="diseases-grid">
                <div className="disease-card">
                  <div className="disease-card-header">
                    <div className="disease-icon-box">🪨</div>
                    <span className="disease-severity-badge">Moderate</span>
                  </div>
                  <h4 className="disease-card-title">Dental Calculus (Tartar)</h4>
                  <p className="disease-card-desc">
                    Calcified bacterial biofilm matrix adhering to tooth enamel and root surfaces. Promotes periodontal breakdown.
                  </p>
                  <p className="disease-card-notes">
                    <strong>Treatment:</strong> Ultrasonic scaling and professional root planing.
                  </p>
                </div>

                <div className="disease-card">
                  <div className="disease-card-header">
                    <div className="disease-icon-box">🦷</div>
                    <span className="disease-severity-badge" style={{ color: '#EF4444', background: '#FEF2F2' }}>
                      High Priority
                    </span>
                  </div>
                  <h4 className="disease-card-title">Dental Caries (Cavity)</h4>
                  <p className="disease-card-desc">
                    Demineralization and structural breakdown of enamel and dentin caused by acidogenic plaque bacteria.
                  </p>
                  <p className="disease-card-notes">
                    <strong>Treatment:</strong> Fluoride remineralization or restorative filling.
                  </p>
                </div>

                <div className="disease-card">
                  <div className="disease-card-header">
                    <div className="disease-icon-box">🩸</div>
                    <span className="disease-severity-badge">Mild / Moderate</span>
                  </div>
                  <h4 className="disease-card-title">Gingivitis</h4>
                  <p className="disease-card-desc">
                    Non-destructive inflammation of the marginal gingiva presenting with edema, redness, and bleeding upon probing.
                  </p>
                  <p className="disease-card-notes">
                    <strong>Treatment:</strong> Oral hygiene instruction, antimicrobial mouth rinses, prophylaxis.
                  </p>
                </div>

                <div className="disease-card">
                  <div className="disease-card-header">
                    <div className="disease-icon-box">🔲</div>
                    <span className="disease-severity-badge">Developmental</span>
                  </div>
                  <h4 className="disease-card-title">Hypodontia</h4>
                  <p className="disease-card-desc">
                    Congenital absence of one or more permanent teeth (excluding third molars), affecting occlusal balance.
                  </p>
                  <p className="disease-card-notes">
                    <strong>Treatment:</strong> Space management, orthodontic alignment, implant restorations.
                  </p>
                </div>

                <div className="disease-card">
                  <div className="disease-card-header">
                    <div className="disease-icon-box">⚪</div>
                    <span className="disease-severity-badge">Acute Discomfort</span>
                  </div>
                  <h4 className="disease-card-title">Oral Ulcer (Aphthae)</h4>
                  <p className="disease-card-desc">
                    Painful mucosal ulceration exposing underlying lamina propria. Typically self-limiting within 10-14 days.
                  </p>
                  <p className="disease-card-notes">
                    <strong>Treatment:</strong> Topical corticosteroids, analgesic gels; biopsy if persistent &gt;14 days.
                  </p>
                </div>

                <div className="disease-card">
                  <div className="disease-card-header">
                    <div className="disease-icon-box">🎨</div>
                    <span className="disease-severity-badge">Aesthetic</span>
                  </div>
                  <h4 className="disease-card-title">Tooth Discoloration</h4>
                  <p className="disease-card-desc">
                    Extrinsic chromogenic deposits or intrinsic enamel alteration from diet, fluorosis, trauma, or tobacco.
                  </p>
                  <p className="disease-card-notes">
                    <strong>Treatment:</strong> Prophylaxis polishing, in-office bleaching, composite bonding or veneers.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* ── Responsible AI Disclaimer ── */}
          <div className="container">
            <div className="disclaimer-banner">
              <ShieldAlert className="disclaimer-icon" size={28} />
              <div className="disclaimer-content">
                <h4>Responsible AI & Clinical Support Disclaimer</h4>
                <p>
                  Dentura AI — Alternative is an artificial intelligence research platform designed solely for clinical screening assistance and exploratory educational decision-support.
                  The model operates without a distinct "healthy/normal" class, meaning ambiguous scans are classified into the closest morphological disease category.
                  This software is NOT a diagnostic medical device and must never substitute a comprehensive physical examination and radiographic assessment by a licensed dental professional.
                </p>
              </div>
            </div>
          </div>
        </main>
      ) : currentView === 'studio' ? (
        /* ── Clinical Analysis Studio View ── */
        <main className="studio-wrapper">
          <div className="container">
            <div className="studio-header" style={{ flexWrap: 'wrap', gap: '1rem' }}>
              <div className="studio-title-box">
                <h1>Oral Pathology Analysis Studio</h1>
                <p>Upload an intraoral photographic scan or choose a benchmark sample image below.</p>
              </div>

              {/* Model Switcher Pills */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', background: '#FFFFFF', padding: '0.25rem', borderRadius: 9999, border: '1px solid #E2E8F0', boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                  <button
                    className={`view-btn ${activeModel === 'efficientnet_b0' ? 'active' : ''}`}
                    onClick={() => setActiveModel('efficientnet_b0')}
                    style={{ fontSize: '0.8rem', padding: '0.4rem 0.85rem' }}
                  >
                    EfficientNetB0 (Deployed)
                  </button>
                  <button
                    className={`view-btn ${activeModel === 'custom_cnn' ? 'active' : ''}`}
                    onClick={() => setActiveModel('custom_cnn')}
                    style={{ fontSize: '0.8rem', padding: '0.4rem 0.85rem' }}
                  >
                    Custom CNN (75.8%)
                  </button>
                  <button
                    className={`view-btn ${activeModel === 'resnet_50' ? 'active' : ''}`}
                    onClick={() => setActiveModel('resnet_50')}
                    style={{ fontSize: '0.8rem', padding: '0.4rem 0.85rem' }}
                  >
                    ResNet50 (73.2%)
                  </button>
                </div>

                {(previewUrl || prediction) && (
                  <button className="btn-secondary" onClick={handleReset} style={{ fontSize: '0.85rem', padding: '0.45rem 1rem' }}>
                    <RotateCcw size={15} /> Reset Studio
                  </button>
                )}
              </div>
            </div>

            {/* Notice when comparative model is selected */}
            {activeModel !== 'efficientnet_b0' && (
              <div
                style={{
                  background: '#F0FDFA',
                  border: '1px solid #99F6E4',
                  borderRadius: 14,
                  padding: '1rem 1.4rem',
                  marginBottom: '1.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '1rem'
                }}
              >
                <div>
                  <div style={{ fontWeight: 700, color: '#0F766E', fontSize: '0.92rem' }}>
                    Comparative Model Selected: {activeModel === 'custom_cnn' ? 'Custom 4-Block CNN' : 'ResNet50 Backbone'}
                  </div>
                  <div style={{ fontSize: '0.82rem', color: '#134E4A', marginTop: '0.2rem' }}>
                    {activeModel === 'custom_cnn'
                      ? 'Custom CNN achieved 75.78% accuracy in research. EfficientNetB0 is the primary trained checkpoint for real-time live inference & Grad-CAM.'
                      : 'ResNet50 achieved 73.18% accuracy in research. EfficientNetB0 is the primary trained checkpoint for real-time live inference & Grad-CAM.'}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    className="btn-secondary"
                    onClick={() => setCurrentView('evaluation')}
                    style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem', background: '#FFFFFF' }}
                  >
                    View Evaluation Metrics
                  </button>
                  <button
                    className="btn-primary"
                    onClick={() => setActiveModel('efficientnet_b0')}
                    style={{ fontSize: '0.8rem', padding: '0.4rem 0.9rem' }}
                  >
                    Switch to Deployed EffNet-B0
                  </button>
                </div>
              </div>
            )}

            {/* Quick Sample Selector */}
            <div className="sample-bar">
              <div className="sample-bar-header">
                <span className="sample-bar-title">
                  <Sparkles size={16} color="#0D9488" /> Try Verified Benchmark Samples
                </span>
                <span style={{ fontSize: '0.8rem', color: '#64748B' }}>Click any sample to test inference & Grad-CAM instantly</span>
              </div>
              <div className="sample-chips-grid">
                {samples.map(sample => (
                  <button
                    key={sample.id}
                    className="sample-chip"
                    onClick={() => handleSampleClick(sample)}
                    disabled={isAnalyzing}
                  >
                    <img src={sample.thumbnail} alt={sample.label} />
                    <div className="sample-chip-info">
                      <span className="sample-chip-name">{sample.label}</span>
                      <span className="sample-chip-sub">{sample.description}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Error Banner */}
            {errorMsg && (
              <div
                style={{
                  background: '#FEF2F2',
                  border: '1px solid #FECACA',
                  borderRadius: 12,
                  padding: '1rem 1.25rem',
                  color: '#991B1B',
                  marginBottom: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem'
                }}
              >
                <AlertTriangle size={20} />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Studio Workspace Grid */}
            <div className="studio-grid">
              {/* Left Column: Upload / Image Preview */}
              <div className="upload-card">
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '1rem', color: '#0A192F' }}>
                  Intraoral Photographic Capture
                </h3>

                {!previewUrl ? (
                  <div
                    className={`dropzone ${isDragOver ? 'drag-active' : ''}`}
                    onDragOver={e => {
                      e.preventDefault()
                      setIsDragOver(true)
                    }}
                    onDragLeave={() => setIsDragOver(false)}
                    onDrop={e => {
                      e.preventDefault()
                      setIsDragOver(false)
                      if (e.dataTransfer.files?.[0]) {
                        handleFileSelect(e.dataTransfer.files[0])
                      }
                    }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <div className="dropzone-icon">
                      <UploadCloud size={28} />
                    </div>
                    <div>
                      <div className="dropzone-title">Select or drag oral scan photo</div>
                      <div className="dropzone-subtitle">Accepts JPG, PNG, WebP (224×224 resolution optimized)</div>
                    </div>
                    <button className="btn-secondary" style={{ marginTop: '0.5rem' }}>
                      Browse Files
                    </button>
                    <input
                      type="file"
                      ref={fileInputRef}
                      style={{ display: 'none' }}
                      accept="image/jpeg,image/png,image/webp"
                      onChange={e => {
                        if (e.target.files?.[0]) handleFileSelect(e.target.files[0])
                      }}
                    />
                  </div>
                ) : (
                  <div>
                    <div className="image-preview-box">
                      <img src={previewUrl} alt="Ingested oral specimen" />
                    </div>

                    <div className="image-actions-bar">
                      <button
                        className="btn-secondary"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isAnalyzing}
                      >
                        Change Image
                      </button>

                      <button
                        className="btn-primary btn-analyze"
                        onClick={handleAnalyze}
                        disabled={isAnalyzing}
                        style={{ marginLeft: '1rem' }}
                      >
                        {isAnalyzing ? (
                          <>
                            <div className="spinner" /> Analyzing Pathology...
                          </>
                        ) : (
                          <>
                            <Microscope size={18} /> Execute Model Analysis
                          </>
                        )}
                      </button>
                    </div>

                    <input
                      type="file"
                      ref={fileInputRef}
                      style={{ display: 'none' }}
                      accept="image/jpeg,image/png,image/webp"
                      onChange={e => {
                        if (e.target.files?.[0]) handleFileSelect(e.target.files[0])
                      }}
                    />
                  </div>
                )}
              </div>

              {/* Right Column: Diagnostic Intelligence */}
              <div className="results-card">
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#0A192F' }}>
                  Clinical Diagnostic Intelligence
                </h3>

                {prediction ? (
                  <>
                    {/* Primary Diagnosis Badge */}
                    <div
                      className="primary-diagnosis-banner"
                      style={{
                        background: prediction.isUncertain ? '#FFFBEB' : '#F0FDFA',
                        borderColor: prediction.isUncertain ? '#FDE68A' : '#99F6E4'
                      }}
                    >
                      <div className="diag-main-info">
                        <div className="diag-icon-large">{prediction.icon}</div>
                        <div>
                          <div className="diag-name">{prediction.displayName}</div>
                          <div className="diag-confidence-label">
                            Severity: <strong>{prediction.diseaseInfo.severity}</strong> • Urgency: {prediction.diseaseInfo.urgency}
                          </div>
                        </div>
                      </div>

                      <div className="diag-score-badge">
                        <div
                          className="diag-score-val"
                          style={{ color: prediction.isUncertain ? '#D97706' : '#0F766E' }}
                        >
                          {(prediction.confidence * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: '0.78rem', color: '#64748B', fontWeight: 600 }}>Confidence</div>
                      </div>
                    </div>

                    {/* Low Confidence Alert */}
                    {prediction.isUncertain && (
                      <div className="uncertain-warning-card">
                        <AlertTriangle size={20} color="#D97706" style={{ flexShrink: 0, marginTop: 2 }} />
                        <p>{prediction.warningMessage}</p>
                      </div>
                    )}

                    {/* Clinical Guidance Box */}
                    <div
                      style={{
                        background: '#F8FAFC',
                        border: '1px solid #E2E8F0',
                        borderRadius: 12,
                        padding: '1.2rem'
                      }}
                    >
                      <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0F172A', marginBottom: '0.35rem' }}>
                        Clinical Description:
                      </div>
                      <p style={{ fontSize: '0.88rem', color: '#334155', margin: 0, marginBottom: '0.75rem', lineHeight: 1.5 }}>
                        {prediction.diseaseInfo.description}
                      </p>
                      <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0F172A', marginBottom: '0.35rem' }}>
                        Management Strategy:
                      </div>
                      <p style={{ fontSize: '0.88rem', color: '#475569', margin: 0, lineHeight: 1.5 }}>
                        {prediction.diseaseInfo.clinicalNotes}
                      </p>
                    </div>

                    {/* Full Probability Breakdown */}
                    <div className="prob-breakdown-card">
                      <div className="prob-title">Categorical Probability Distribution (All 6 Classes)</div>
                      <div className="prob-list">
                        {prediction.probabilities.map(item => (
                          <div key={item.classKey} className="prob-item-row">
                            <span className="prob-item-label">
                              <span>{item.icon}</span> {item.displayName}
                            </span>
                            <div className="prob-track-outer">
                              <div
                                className="prob-fill-inner"
                                style={{
                                  width: `${Math.max(item.probability * 100, 2)}%`,
                                  background: item.badgeColor
                                }}
                              />
                            </div>
                            <span className="prob-percentage-text">{(item.probability * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <div
                    style={{
                      textAlign: 'center',
                      padding: '4rem 2rem',
                      background: '#F8FAFC',
                      borderRadius: 14,
                      border: '1px dashed #CBD5E1',
                      color: '#64748B'
                    }}
                  >
                    <Microscope size={48} color="#94A3B8" style={{ margin: '0 auto 1rem' }} />
                    <h4 style={{ fontSize: '1.15rem', color: '#0F172A', fontWeight: 700, margin: '0 0 0.5rem 0' }}>
                      Awaiting Image Ingestion
                    </h4>
                    <p style={{ fontSize: '0.9rem', margin: 0, maxWidth: 360, marginInline: 'auto' }}>
                      Upload an intraoral photograph or pick one of the verified sample scans above to run real-time classification.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Explainability / Grad-CAM Section */}
            {gradcam && gradcam.supported && (
              <div className="explain-card">
                <div className="explain-header">
                  <div className="explain-header-left">
                    <h3>
                      <Eye size={22} color="#0D9488" /> Grad-CAM Spatial Explainability
                    </h3>
                    <p>
                      Visualizing gradients backpropagated to layer <code>{gradcam.targetLayer}</code> to verify anatomical lesion focus zones.
                    </p>
                  </div>

                  <div className="view-switcher">
                    <button
                      className={`view-btn ${gradcamViewMode === 'all' ? 'active' : ''}`}
                      onClick={() => setGradcamViewMode('all')}
                    >
                      Tri-View Comparison
                    </button>
                    <button
                      className={`view-btn ${gradcamViewMode === 'overlay' ? 'active' : ''}`}
                      onClick={() => setGradcamViewMode('overlay')}
                    >
                      Overlay Only
                    </button>
                    <button
                      className={`view-btn ${gradcamViewMode === 'heatmap' ? 'active' : ''}`}
                      onClick={() => setGradcamViewMode('heatmap')}
                    >
                      Thermal Map
                    </button>
                    <button
                      className={`view-btn ${gradcamViewMode === 'original' ? 'active' : ''}`}
                      onClick={() => setGradcamViewMode('original')}
                    >
                      Original
                    </button>
                  </div>
                </div>

                {gradcamViewMode === 'all' ? (
                  <div className="gradcam-display-container">
                    <div className="gradcam-tile">
                      <img src={gradcam.originalBase64} alt="Preprocessed input" className="gradcam-tile-img" />
                      <div className="gradcam-tile-caption">
                        <span>1. Preprocessed Input</span>
                        <span style={{ fontSize: '0.75rem', color: '#64748B' }}>224 × 224 RGB</span>
                      </div>
                    </div>

                    <div className="gradcam-tile">
                      <img src={gradcam.heatmapBase64} alt="Jet thermal heatmap" className="gradcam-tile-img" />
                      <div className="gradcam-tile-caption">
                        <span>2. Thermal Attention (Jet)</span>
                        <span style={{ fontSize: '0.75rem', color: '#64748B' }}>Gradient Intensities</span>
                      </div>
                    </div>

                    <div className="gradcam-tile">
                      <img src={gradcam.overlayBase64} alt="Blended lesion overlay" className="gradcam-tile-img" />
                      <div className="gradcam-tile-caption">
                        <span>3. Blended Overlay</span>
                        <span className="clinical-badge" style={{ fontSize: '0.65rem' }}>
                          Diagnostic Focus
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ maxWidth: 520, margin: '0 auto' }}>
                    <div className="gradcam-tile">
                      <img
                        src={
                          gradcamViewMode === 'overlay'
                            ? gradcam.overlayBase64
                            : gradcamViewMode === 'heatmap'
                            ? gradcam.heatmapBase64
                            : gradcam.originalBase64
                        }
                        alt="Selected Grad-CAM representation"
                        style={{ width: '100%', height: 380, objectFit: 'cover' }}
                      />
                      <div className="gradcam-tile-caption">
                        <span>
                          {gradcamViewMode === 'overlay'
                            ? 'Grad-CAM Attention Overlay'
                            : gradcamViewMode === 'heatmap'
                            ? 'Jet Thermal Gradient Heatmap'
                            : 'Standard Preprocessed Input'}
                        </span>
                        <span style={{ fontSize: '0.8rem', color: '#0D9488', fontWeight: 700 }}>
                          {gradcam.predictedClass.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      ) : (
        <EvaluationDashboard />
      )}

      {/* ── Footer ── */}
      <footer className="app-footer">
        <div className="container">
          <div className="footer-top">
            <div className="footer-brand">
              <div className="nav-brand-icon" style={{ width: 32, height: 32 }}>
                <Stethoscope size={18} />
              </div>
              <span>Dentura AI — Alternative</span>
            </div>
            <div className="footer-meta">
              Based on Abdelhameed Nasr's Oral Disease Detector • EfficientNetB0 Transfer Learning
            </div>
          </div>

          <div className="footer-bottom">
            <span>Clinical AI Decision-Support Platform • Standalone Implementation</span>
            <span>Isolated Environment: /home/sid/Dentura_2</span>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
