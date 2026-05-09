'use client';

import { TrainingRun, TrainingStatus } from '@/lib/types';
import { Activity, History, Clock, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  activeRunId: string | null;
  trainingStatus: TrainingStatus | null;
  runs: TrainingRun[];
  onSelectRun: (runId: string) => void;
  onClose: () => void;
}

export function TrainingPanel({ activeRunId, trainingStatus, runs, onSelectRun, onClose }: Props) {
  const [tab, setTab] = useState<'active' | 'history'>('active');

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="fixed inset-x-0 bottom-0 bg-lab-surface border-t border-lab-border shadow-2xl z-50 max-h-[50vh] flex flex-col"
    >
      <div className="flex items-center justify-between px-4 py-2 border-b border-lab-border">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setTab('active')}
            className={`flex items-center gap-2 px-3 py-1.5 text-sm font-mono rounded transition-colors ${
              tab === 'active' ? 'bg-lab-accentDim text-lab-accent' : 'text-lab-textMuted hover:text-lab-text'
            }`}
          >
            <Activity className="w-4 h-4" />
            ACTIVE RUN
          </button>
          <button
            onClick={() => setTab('history')}
            className={`flex items-center gap-2 px-3 py-1.5 text-sm font-mono rounded transition-colors ${
              tab === 'history' ? 'bg-lab-accentDim text-lab-accent' : 'text-lab-textMuted hover:text-lab-text'
            }`}
          >
            <History className="w-4 h-4" />
            HISTORY
          </button>
        </div>
        <button onClick={onClose} className="text-lab-textMuted hover:text-lab-text text-sm">CLOSE</button>
      </div>

      <div className="flex-1 overflow-auto p-4 scrollbar-thin">
        <AnimatePresence mode="wait">
          {tab === 'active' && activeRunId && (
            <motion.div
              key="active"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-3"
            >
              {trainingStatus ? (
                <>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm text-lab-accent">{trainingStatus.run_id}</span>
                    <span className={`font-mono text-xs px-2 py-1 rounded ${
                      trainingStatus.status === 'running' ? 'bg-lab-yellowDim text-lab-yellow' :
                      trainingStatus.status === 'completed' ? 'bg-lab-greenDim text-lab-green' :
                      'bg-lab-redDim text-lab-red'
                    }`}>
                      {trainingStatus.status.toUpperCase()}
                    </span>
                  </div>
                  
                  <div className="w-full bg-lab-bg rounded-full h-2">
                    <div 
                      className="bg-lab-accent h-2 rounded-full transition-all"
                      style={{ width: `${(trainingStatus.current_step / trainingStatus.total_steps) * 100}%` }}
                    />
                  </div>
                  
                  <p className="font-mono text-xs text-lab-textMuted">
                    Step {trainingStatus.current_step} / {trainingStatus.total_steps}
                  </p>

                  <div className="bg-lab-bg rounded-lg p-3 font-mono text-xs space-y-1 max-h-48 overflow-auto">
                    {trainingStatus.logs.slice(-20).map((log, i) => (
                      <p key={i} className="text-lab-textMuted">{log}</p>
                    ))}
                    {trainingStatus.status === 'running' && (
                      <motion.span
                        animate={{ opacity: [0, 1, 0] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                        className="text-lab-accent"
                      >
                        _
                      </motion.span>
                    )}
                  </div>
                </>
              ) : (
                <p className="text-lab-textMuted text-sm">No active training run.</p>
              )}
            </motion.div>
          )}

          {tab === 'history' && (
            <motion.div
              key="history"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-2"
            >
              {runs.length === 0 ? (
                <p className="text-lab-textMuted text-sm">No training runs yet.</p>
              ) : (
                runs.map(run => (
                  <button
                    key={run.run_id}
                    onClick={() => onSelectRun(run.run_id)}
                    className="w-full flex items-center justify-between p-3 bg-lab-bg border border-lab-border rounded-lg hover:border-lab-accent transition-colors text-left"
                  >
                    <div className="flex items-center gap-3">
                      <Clock className="w-4 h-4 text-lab-textMuted" />
                      <div>
                        <p className="font-mono text-sm text-lab-text">{run.run_id}</p>
                        <p className="font-mono text-xs text-lab-textMuted">
                          {new Date(run.timestamp).toLocaleDateString()} · {run.num_steps} steps
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className="font-mono text-xs text-lab-green">ACC {(run.final_accuracy * 100).toFixed(0)}%</p>
                        <p className="font-mono text-xs text-lab-accent">TTS {run.final_mean_tts.toFixed(3)}</p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-lab-textMuted" />
                    </div>
                  </button>
                ))
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
