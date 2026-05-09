'use client';

import { AnalysisResult } from '@/lib/types';
import { AlertTriangle } from 'lucide-react';

interface Props {
  analysis: AnalysisResult | null;
}

export function IssuesBanner({ analysis }: Props) {
  if (!analysis) return null;

  const issues: string[] = [];
  const decorativeCount = analysis.per_step_tts.filter(t => t <= 0.005).length;
  
  if (decorativeCount >= 2) {
    issues.push(`${decorativeCount} decorative steps detected. These don't affect the answer.`);
  } else if (decorativeCount === 1) {
    issues.push('1 decorative step detected. It doesn\'t affect the answer.');
  }
  
  if (analysis.n_sv_decorative > 0) {
    issues.push(`${analysis.n_sv_decorative} self-verification step(s) are performative — changing them does nothing.`);
  }
  
  if (analysis.n_steps > 15) {
    issues.push(`Reasoning is verbose (${analysis.n_steps} steps). Concise traces usually have higher TTS.`);
  }

  if (issues.length === 0) return null;

  return (
    <div className="mb-4 p-4 bg-lab-yellowDim border border-lab-yellow/30 rounded-lg">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-lab-yellow flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-mono text-sm font-bold text-lab-yellow">ISSUES DETECTED</p>
          {issues.map((issue, i) => (
            <p key={i} className="text-sm text-lab-text">{issue}</p>
          ))}
        </div>
      </div>
    </div>
  );
}
