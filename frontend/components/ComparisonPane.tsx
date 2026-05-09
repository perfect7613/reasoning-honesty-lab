'use client';

import { AnalysisResult } from '@/lib/types';
import { motion } from 'framer-motion';
import { ArrowRight, Download } from 'lucide-react';

interface Props {
  baseline: AnalysisResult | null;
  improved: AnalysisResult | null;
  onExport: () => void;
}

function MiniHeatmap({ analysis, label }: { analysis: AnalysisResult | null; label: string }) {
  if (!analysis) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-lab-textMuted tracking-wider">{label}</span>
        <span className="font-mono text-xs text-lab-textMuted">{analysis.n_steps} steps</span>
      </div>
      
      <div className="flex flex-wrap gap-1">
        {analysis.per_step_tts.map((tts, i) => (
          <motion.div
            key={i}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: i * 0.01 }}
            className={`w-3 h-8 rounded-sm ${
              tts >= 0.7 ? 'bg-lab-green' : tts <= 0.005 ? 'bg-lab-red' : 'bg-lab-yellow'
            }`}
            title={`Step ${i + 1}: TTS = ${tts.toFixed(3)}`}
          />
        ))}
      </div>

      <div className="grid grid-cols-3 gap-2 mt-2">
        <div className="text-center">
          <p className="font-mono text-lg font-bold text-lab-green">{(analysis.frac_high_tts * 100).toFixed(0)}%</p>
          <p className="font-mono text-[10px] text-lab-textMuted">HIGH TTS</p>
        </div>
        <div className="text-center">
          <p className="font-mono text-lg font-bold text-lab-accent">{analysis.mean_tts.toFixed(3)}</p>
          <p className="font-mono text-[10px] text-lab-textMuted">MEAN</p>
        </div>
        <div className="text-center">
          <p className="font-mono text-lg font-bold text-lab-red">{(analysis.frac_decorative * 100).toFixed(0)}%</p>
          <p className="font-mono text-[10px] text-lab-textMuted">DECORATIVE</p>
        </div>
      </div>
    </div>
  );
}

export function ComparisonPane({ baseline, improved, onExport }: Props) {
  if (!baseline || !improved) return null;

  const stepReduction = ((baseline.n_steps - improved.n_steps) / baseline.n_steps * 100).toFixed(0);
  const ttsImprovement = ((improved.mean_tts - baseline.mean_tts) / baseline.mean_tts * 100).toFixed(0);
  const decorativeReduction = ((baseline.frac_decorative - improved.frac_decorative) / baseline.frac_decorative * 100).toFixed(0);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-lab-surface border border-lab-border rounded-lg p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="font-mono text-lg font-bold text-lab-text">BEFORE / AFTER</h2>
        <button
          onClick={onExport}
          className="flex items-center gap-2 px-3 py-1.5 bg-lab-bg border border-lab-border rounded text-sm font-mono text-lab-textMuted hover:text-lab-text hover:border-lab-accent transition-colors"
        >
          <Download className="w-4 h-4" />
          EXPORT
        </button>
      </div>

      <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-start">
        <MiniHeatmap analysis={baseline} label="BASELINE" />
        
        <div className="flex flex-col items-center gap-2 pt-8">
          <ArrowRight className="w-5 h-5 text-lab-accent" />
          <div className="space-y-2 text-center">
            <div className="bg-lab-greenDim px-2 py-1 rounded">
              <p className="font-mono text-xs text-lab-green font-bold">-{stepReduction}%</p>
              <p className="font-mono text-[10px] text-lab-textMuted">STEPS</p>
            </div>
            <div className="bg-lab-accentDim px-2 py-1 rounded">
              <p className="font-mono text-xs text-lab-accent font-bold">+{ttsImprovement}%</p>
              <p className="font-mono text-[10px] text-lab-textMuted">TTS</p>
            </div>
            <div className="bg-lab-greenDim px-2 py-1 rounded">
              <p className="font-mono text-xs text-lab-green font-bold">-{decorativeReduction}%</p>
              <p className="font-mono text-[10px] text-lab-textMuted">DECORATIVE</p>
            </div>
          </div>
        </div>

        <MiniHeatmap analysis={improved} label="TRAINED" />
      </div>
    </motion.div>
  );
}
