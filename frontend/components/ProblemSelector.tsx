'use client';

import { Problem } from '@/lib/types';
import { ChevronDown, Terminal } from 'lucide-react';
import { useState } from 'react';
import { MathText } from './MathText';

interface Props {
  problems: Problem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onFixReasoning: () => void;
  isTraining: boolean;
}

export function ProblemSelector({ problems, selectedId, onSelect, onFixReasoning, isTraining }: Props) {
  const [open, setOpen] = useState(false);
  const selected = problems.find(p => p.id === selectedId);

  return (
    <div className="flex items-center gap-3 mb-6">
      <div className="relative flex-1">
        <button
          onClick={() => setOpen(!open)}
          className="w-full flex items-center justify-between px-4 py-3 bg-lab-surface border border-lab-border rounded-lg text-left hover:border-lab-accent transition-colors"
        >
          <span className="flex items-center gap-2 min-w-0">
            <Terminal className="w-4 h-4 text-lab-accent flex-shrink-0" />
            <span className="text-sm text-lab-text truncate">
              {selected ? (
                <MathText text={selected.question} maxLength={60} />
              ) : 'Select a problem...'}
            </span>
          </span>
          <ChevronDown className={`w-4 h-4 text-lab-textMuted transition-transform flex-shrink-0 ${open ? 'rotate-180' : ''}`} />
        </button>
        
        {open && (
          <div className="absolute z-50 mt-1 w-full max-h-64 overflow-auto bg-lab-surface border border-lab-border rounded-lg shadow-2xl">
            {problems.map(problem => (
              <button
                key={problem.id}
                onClick={() => { onSelect(problem.id); setOpen(false); }}
                className={`w-full px-4 py-2 text-left text-sm hover:bg-lab-surfaceHover transition-colors ${
                  selectedId === problem.id ? 'bg-lab-accentDim text-lab-accent' : 'text-lab-text'
                }`}
              >
                <span className="font-mono text-xs text-lab-textMuted mr-2">{problem.id}</span>
                <MathText text={problem.question} maxLength={70} />
              </button>
            ))}
          </div>
        )}
      </div>

      <button
        onClick={onFixReasoning}
        disabled={isTraining || !selectedId}
        className="px-5 py-3 bg-lab-accent text-lab-bg font-mono text-sm font-bold rounded-lg hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex-shrink-0"
      >
        {isTraining ? 'TRAINING...' : 'FIX REASONING'}
      </button>
    </div>
  );
}
