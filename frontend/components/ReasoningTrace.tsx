'use client';

import { MathText } from './MathText';

interface Props {
  cotText: string;
  perStepTts: number[];
}

export function ReasoningTrace({ cotText, perStepTts }: Props) {
  // Split by step boundaries (similar to backend segmentation)
  const steps = cotText.split(/\n(?=\d+[\.\)]\s|Step\s\d|First|Second|Third|Next|Then|Now|Finally|So\s|Therefore|Thus|Hence|Let me|Let's|Wait|Hmm|Actually|We\s|I\s|This\s|Since\s|Because\s|Given\s|If\s|But\s|However\s|Note\s|For\s|Using\s|Substitut|Comput|Calculat|Evaluat|Apply|Recall)/i);

  return (
    <div className="mb-6 border border-lab-border rounded-lg overflow-hidden">
      <div className="px-4 py-2 bg-lab-surface border-b border-lab-border">
        <span className="text-xs font-mono text-lab-accent uppercase tracking-wider">
          Generated Reasoning Trace
        </span>
      </div>
      <div className="p-4 space-y-2 max-h-96 overflow-auto">
        {steps.map((step, i) => {
          const tts = perStepTts[i] ?? 0;
          const isDecorative = tts <= 0.005;
          const isHigh = tts >= 0.7;
          
          return (
            <div 
              key={i} 
              className={`p-2 rounded border-l-2 text-sm ${
                isDecorative 
                  ? 'border-red-500 bg-red-500/5' 
                  : isHigh 
                    ? 'border-green-500 bg-green-500/5'
                    : 'border-yellow-500 bg-yellow-500/5'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-mono text-lab-textMuted">
                  Step {i + 1}
                </span>
                <span className={`text-xs font-mono font-bold ${
                  isDecorative ? 'text-red-400' : isHigh ? 'text-green-400' : 'text-yellow-400'
                }`}>
                  TTS: {tts.toFixed(4)}
                </span>
              </div>
              <MathText text={step.trim()} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
