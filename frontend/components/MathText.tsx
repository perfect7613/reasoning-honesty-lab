'use client';

import { InlineMath } from 'react-katex';

interface Props {
  text: string;
  className?: string;
  maxLength?: number;
}

export function MathText({ text, className = '', maxLength }: Props) {
  // Truncate if needed
  const displayText = maxLength && text.length > maxLength 
    ? text.substring(0, maxLength) + '...' 
    : text;

  // Split by inline math delimiters $...$
  const parts = displayText.split(/(\$[^$]+\$)/g);

  return (
    <span className={className}>
      {parts.map((part, i) => {
        if (part.startsWith('$') && part.endsWith('$')) {
          const math = part.slice(1, -1);
          try {
            return <InlineMath key={i} math={math} />;
          } catch {
            return <span key={i}>{part}</span>;
          }
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}
