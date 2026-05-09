export interface Problem {
  id: string;
  question: string;
  answer: string;
}

export interface StepTTS {
  step_index: number;
  step_text: string;
  tts: number;
  is_self_verification: boolean;
}

export interface AnalysisResult {
  id?: string;
  question: string;
  answer: string;
  model_correct: boolean;
  n_steps: number;
  mean_tts: number;
  frac_high_tts: number;
  frac_decorative: number;
  n_self_verification: number;
  n_sv_decorative: number;
  per_step_tts: number[];
  cot_text?: string;
}

export interface TrainingRun {
  run_id: string;
  status: string;
  timestamp: string;
  num_steps: number;
  final_accuracy: number;
  final_mean_tts: number;
  final_mean_steps: number;
  final_decorative: number;
}

export interface TrainingStatus {
  run_id: string;
  status: string;
  current_step: number;
  total_steps: number;
  latest_metrics: Record<string, any>;
  logs: string[];
}

export interface ComparisonData {
  run_id: string;
  baseline: { analyses: AnalysisResult[] } | null;
  improved: { analyses: AnalysisResult[] } | null;
}
