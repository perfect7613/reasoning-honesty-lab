import { Problem, AnalysisResult, TrainingRun, TrainingStatus, ComparisonData } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  getProblems: () => fetchAPI<Problem[]>('/problems'),
  
  analyze: (problemId: string, mode: 'baseline' | 'live' = 'baseline') =>
    fetchAPI<AnalysisResult>('/analyze', {
      method: 'POST',
      body: JSON.stringify({ problem_id: problemId, mode }),
    }),
  
  startTraining: (problemIds?: string[], numSteps = 20, groupSize = 8) =>
    fetchAPI<{ run_id: string; status: string }>('/train/start', {
      method: 'POST',
      body: JSON.stringify({ problem_ids: problemIds, num_steps: numSteps, group_size: groupSize }),
    }),
  
  getTrainingStatus: (runId: string) =>
    fetchAPI<TrainingStatus>(`/train/status/${runId}`),
  
  getRuns: () => fetchAPI<TrainingRun[]>('/runs'),
  
  getRun: (runId: string) => fetchAPI<ComparisonData>(`/runs/${runId}`),
  
  exportAnalysis: (analysisId: string) =>
    fetchAPI<Record<string, any>>(`/export/${analysisId}`),
};
