'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ScanLine, Zap, BrainCircuit, Radio, Database } from 'lucide-react';
import { api } from '@/lib/api';
import { Problem, AnalysisResult, TrainingRun, TrainingStatus, ComparisonData } from '@/lib/types';
import { ProblemSelector } from '@/components/ProblemSelector';
import { StepHeatmap } from '@/components/StepHeatmap';
import { IssuesBanner } from '@/components/IssuesBanner';
import { MetricStrip } from '@/components/MetricStrip';
import { TrainingPanel } from '@/components/TrainingPanel';
import { ComparisonPane } from '@/components/ComparisonPane';
import { ReasoningTrace } from '@/components/ReasoningTrace';

export default function Home() {
  const [problems, setProblems] = useState<Problem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [liveMode, setLiveMode] = useState(true);
  const [analysisStage, setAnalysisStage] = useState('');
  
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [showPanel, setShowPanel] = useState(false);
  
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [comparison, setComparison] = useState<ComparisonData | null>(null);

  // Load problems on mount
  useEffect(() => {
    api.getProblems().then(setProblems).catch(console.error);
    api.getRuns().then(setRuns).catch(console.error);
  }, []);

  // Poll training status
  useEffect(() => {
    if (!activeRunId || !isTraining) return;
    
    const interval = setInterval(async () => {
      try {
        const status = await api.getTrainingStatus(activeRunId);
        setTrainingStatus(status);
        
        if (status.status === 'completed' || status.status === 'failed') {
          setIsTraining(false);
          api.getRuns().then(setRuns);
        }
      } catch (e) {
        console.error(e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeRunId, isTraining]);

  const handleAnalyze = useCallback(async (problemId: string) => {
    setLoading(true);
    setComparison(null);
    setAnalysisStage(liveMode ? 'Generating chain-of-thought with DeepSeek-V3.1...' : 'Loading cached analysis...');
    
    try {
      const result = await api.analyze(problemId, liveMode ? 'live' : 'baseline');
      setAnalysis(result);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
      setAnalysisStage('');
    }
  }, [liveMode]);

  const handleFixReasoning = useCallback(async () => {
    if (!selectedId) return;
    
    setIsTraining(true);
    setShowPanel(true);
    setComparison(null);
    
    try {
      // Pass ALL problems for training, not just the selected one.
      // GRPO needs variance within the group — a single problem gives n=1, std=0.
      const allProblemIds = problems.map(p => p.id);
      const result = await api.startTraining(allProblemIds, 20, 8);
      setActiveRunId(result.run_id);
      setTrainingStatus({
        run_id: result.run_id,
        status: 'running',
        current_step: 0,
        total_steps: 20,
        latest_metrics: {},
        logs: ['Training started...'],
      });
    } catch (e) {
      console.error(e);
      setIsTraining(false);
    }
  }, [selectedId, problems]);

  const handleSelectRun = useCallback(async (runId: string) => {
    try {
      const data = await api.getRun(runId);
      setComparison(data);
      setShowPanel(false);
      
      // Find matching analysis for the selected problem
      if (selectedId && data.improved) {
        const improved = data.improved.analyses.find(a => a.id === selectedId);
        if (improved) {
          setAnalysis(improved);
        }
      }
    } catch (e) {
      console.error(e);
    }
  }, [selectedId]);

  const handleExport = useCallback(() => {
    if (!analysis) return;
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `reasoning-receipt-${selectedId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [analysis, selectedId]);

  return (
    <main className="min-h-screen p-6 max-w-6xl mx-auto">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <ScanLine className="w-8 h-8 text-lab-accent" />
          <h1 className="text-3xl font-display font-bold text-lab-text tracking-tight">
            REASONING HONESTY LAB
          </h1>
        </div>
        <p className="text-lab-textMuted text-sm max-w-xl">
          An X-ray for AI reasoning. Measures which steps actually change the answer, 
          then fine-tunes models for more causally faithful thinking.
        </p>
        <div className="flex items-center gap-4 mt-3">
          <span className="flex items-center gap-1.5 text-xs font-mono text-lab-textMuted">
            <BrainCircuit className="w-3 h-3" />
            deepseek-ai/DeepSeek-V3.1
          </span>
          <span className="flex items-center gap-1.5 text-xs font-mono text-lab-textMuted">
            <Zap className="w-3 h-3" />
            TTS-powered RL
          </span>
        </div>
      </header>

      {/* Mode Toggle */}
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={() => setLiveMode(true)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-colors ${
            liveMode 
              ? 'bg-lab-accent text-lab-bg' 
              : 'bg-lab-surface text-lab-textMuted hover:text-lab-text'
          }`}
        >
          <Radio className="w-3 h-3" />
          LIVE
        </button>
        <button
          onClick={() => setLiveMode(false)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-colors ${
            !liveMode 
              ? 'bg-lab-accent text-lab-bg' 
              : 'bg-lab-surface text-lab-textMuted hover:text-lab-text'
          }`}
        >
          <Database className="w-3 h-3" />
          CACHED
        </button>
        <span className="text-xs text-lab-textMuted ml-2">
          {liveMode ? 'Model generates reasoning live (~2 min)' : 'Instant precomputed results'}
        </span>
      </div>

      {/* Problem Selector */}
      <ProblemSelector
        problems={problems}
        selectedId={selectedId}
        onSelect={(id) => { setSelectedId(id); handleAnalyze(id); }}
        onFixReasoning={handleFixReasoning}
        isTraining={isTraining}
      />

      {/* Loading */}
      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col gap-3 py-8"
        >
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 border-2 border-lab-accent border-t-transparent rounded-full animate-spin" />
            <span className="font-mono text-sm text-lab-textMuted">{analysisStage}</span>
          </div>
          {liveMode && (
            <div className="text-xs text-lab-textMuted font-mono space-y-1 pl-7">
              <p>1. Generating chain-of-thought...</p>
              <p>2. Segmenting into reasoning steps...</p>
              <p>3. Computing TTS for each step (4 API calls per step)...</p>
              <p>4. Building heatmap and diagnostics...</p>
            </div>
          )}
        </motion.div>
      )}

      {/* Analysis Results */}
      <AnimatePresence>
        {analysis && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <IssuesBanner analysis={analysis} />
            <MetricStrip analysis={analysis} />
            
            {/* Generated Reasoning Trace */}
            {analysis.cot_text && (
              <ReasoningTrace 
                cotText={analysis.cot_text} 
                perStepTts={analysis.per_step_tts}
              />
            )}
            
            <StepHeatmap analysis={analysis} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Comparison Pane */}
      <AnimatePresence>
        {comparison && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6"
          >
            <ComparisonPane
              baseline={comparison.baseline?.analyses.find(a => a.id === selectedId) || null}
              improved={comparison.improved?.analyses.find(a => a.id === selectedId) || null}
              onExport={handleExport}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Training Panel */}
      <AnimatePresence>
        {showPanel && (
          <TrainingPanel
            activeRunId={activeRunId}
            trainingStatus={trainingStatus}
            runs={runs}
            onSelectRun={handleSelectRun}
            onClose={() => setShowPanel(false)}
          />
        )}
      </AnimatePresence>

      {/* Empty State */}
      {!analysis && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center justify-center py-20 text-center"
        >
          <ScanLine className="w-16 h-16 text-lab-border mb-4" />
          <p className="text-lab-textMuted text-sm mb-2">Select a problem to begin analysis</p>
          <p className="text-lab-textMuted text-xs font-mono">20 curated MATH-500 problems available</p>
        </motion.div>
      )}
    </main>
  );
}
