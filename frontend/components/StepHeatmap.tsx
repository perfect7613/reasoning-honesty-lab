'use client';

import { AnalysisResult } from '@/lib/types';
import { motion } from 'framer-motion';

interface Props {
  analysis: AnalysisResult | null;
}

function getStepColor(tts: number): string {
  if (tts >= 0.7) return 'border-lab-green bg-lab-greenDim';
  if (tts <= 0.005) return 'border-lab-red bg-lab-redDim';
  return 'border-lab-yellow bg-lab-yellowDim';
}

function getStepBadge(tts: number): string {
  if (tts >= 0.7) return 'text-lab-green';
  if (tts <= 0.005) return 'text-lab-red';
  return 'text-lab-yellow';
}

export function StepHeatmap({ analysis }: Props) {
  if (!analysis) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-4 mb-3">
        <span className="font-mono text-[10px] text-lab-textMuted tracking-wider">REASONING TRACE</span>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-[10px] text-lab-green"><span className="w-2 h-2 rounded-full bg-lab-green"></span>TRUE</span>
          <span className="flex items-center gap-1 text-[10px] text-lab-yellow"><span className="w-2 h-2 rounded-full bg-lab-yellow"></span>PARTIAL</span>
          <span className="flex items-center gap-1 text-[10px] text-lab-red"><span className="w-2 h-2 rounded-full bg-lab-red"></span>DECORATIVE</span>
        </div>
      </div>

      {analysis.per_step_tts.map((tts, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.03, duration: 0.3 }}
          className={`flex items-center gap-3 p-3 border rounded-lg ${getStepColor(tts)}`}
        >
          <span className="font-mono text-xs text-lab-textMuted w-6">{i + 1}</span>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-lab-text truncate">
              Step {i + 1}: {tts >= 0.7 ? 'Causally important' : tts <= 0.005 ? 'Decorative filler' : 'Partial contribution'}
            </p>
          </div>
          <span className={`font-mono text-sm font-bold ${getStepBadge(tts)}`}>
            {tts.toFixed(3)}
          </span>
        </motion.div>
      ))}
    </div>
  );
}
