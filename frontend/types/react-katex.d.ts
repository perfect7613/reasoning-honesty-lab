declare module 'react-katex' {
  import { ReactNode } from 'react';
  
  interface MathProps {
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
  }
  
  export function InlineMath(props: MathProps): ReactNode;
  export function BlockMath(props: MathProps): ReactNode;
}
