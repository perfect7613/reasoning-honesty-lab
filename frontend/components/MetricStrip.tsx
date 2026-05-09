'use client';

import { AnalysisResult } from '@/lib/types';
import { CheckCircle2, XCircle, Activity, Layers, AlertCircle } from 'lucide-react';

interface Props {
  analysis: AnalysisResult | null;
}

export function MetricStrip({ analysis }: Props) {
  if (!analysis) return null;

  const metrics = [
    {
      label: 'ACCURACY',
      value: analysis.model_correct ? '100%' : '0%',
      icon: analysis.model_correct ? CheckCircle2 : XCircle,
      color: analysis.model_correct ? 'text-lab-green' : 'text-lab-red',
    },
    {
      label: 'MEAN TTS',
      value: analysis.mean_tts.toFixed(3),
      icon: Activity,
      color: 'text-lab-accent',
    },
    {
      label: 'STEPS',
      value: String(analysis.n_steps),
      icon: Layers,
      color: 'text-lab-text',
    },
    {
      label: 'DECORATIVE',
      value: `${(analysis.frac_decorative * 100).toFixed(0)}%`,
      icon: AlertCircle,
      color: analysis.frac_decorative > 0.3 ? 'text-lab-red' : 'text-lab-textMuted',
    },
  ];

  return (
    <div className="grid grid-cols-4 gap-3 mb-6">
      {metrics.map(m => (
        <div key={m.label} className="bg-lab-surface border border-lab-border rounded-lg p-3">
          <div className="flex items-center gap-2 mb-1">
            <m.icon className={`w-4 h-4 ${m.color}`} />
            <span className="font-mono text-[10px] text-lab-textMuted tracking-wider">{m.label}</span>
          </div>
          <p className={`font-mono text-xl font-bold ${m.color}`}>{m.value}</p>
        </div>
      ))}
    </div>
  );
}
