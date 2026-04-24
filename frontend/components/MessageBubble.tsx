'use client';

import { BookOpen, FileText, ShieldOff } from 'lucide-react';
import { Citation } from '@/lib/types';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Citation[];
  isGrounded?: boolean;
  retrievalScore?: number;
  timestamp: Date;
  isError?: boolean;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function renderContent(text: string) {
  // Simple markdown-like rendering for bold, newlines, and lists
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];

  lines.forEach((line, i) => {
    if (!line.trim()) {
      elements.push(<br key={`br-${i}`} />);
      return;
    }

    // Bold with **text**
    const parts = line.split(/(\*\*[^*]+\*\*)/g);
    const rendered = parts.map((part, j) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={j}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });

    if (line.match(/^[-•]\s/)) {
      elements.push(
        <li key={i} style={{ marginBottom: 3 }}>
          {rendered}
        </li>,
      );
    } else {
      elements.push(<p key={i}>{rendered}</p>);
    }
  });

  return <div className="message-content">{elements}</div>;
}

function confidenceLabel(score: number): string {
  if (score >= 0.7) return 'High';
  if (score >= 0.45) return 'Medium';
  return 'Low';
}

function confidenceColor(score: number): string {
  if (score >= 0.7) return 'var(--success)';
  if (score >= 0.45) return 'var(--warning)';
  return '#f87171';
}

export default function MessageBubble({
  role,
  content,
  sources,
  isGrounded,
  retrievalScore,
  timestamp,
  isError,
}: MessageBubbleProps) {
  const hasCitations = sources && sources.length > 0;
  // isGrounded is undefined for user messages — treat undefined as true
  const isRefusal = role === 'assistant' && isGrounded === false;
  const isMultiPdf = hasCitations
    ? new Set(sources.map((s) => s.pdf_name)).size > 1
    : false;

  return (
    <div className={`message-wrapper ${role}`}>
      {/* Avatar + time row */}
      <div
        className="message-meta"
        style={{ flexDirection: role === 'user' ? 'row-reverse' : 'row' }}
      >
        {role === 'assistant' && (
          <div className={`message-avatar ${role}`}>
            {isRefusal ? <ShieldOff size={14} /> : <FileText size={14} />}
          </div>
        )}
        <span className="message-time">{formatTime(timestamp)}</span>
      </div>

      {/* Bubble + (optional) citation block */}
      <div style={{ maxWidth: '72%', display: 'flex', flexDirection: 'column' }}>
        <div
          className={`message-bubble ${role} ${isError ? 'error-bubble' : ''} ${isRefusal ? 'refusal-bubble' : ''} ${hasCitations ? 'has-citations' : ''}`}
        >
          {renderContent(content)}
        </div>

        {/* Citations */}
        {hasCitations && (
          <div className="citations-container">
            <span className="citations-label">
              <BookOpen size={11} />
              Sources:
            </span>
            {sources.map((c, idx) => (
              <span key={idx} className="citation-chip" title={c.pdf_name}>
                <FileText size={9} />
                p.{c.page_number}
                {isMultiPdf && (
                  <span
                    style={{
                      opacity: 0.7,
                      maxWidth: 80,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    &nbsp;— {c.pdf_name.replace('.pdf', '')}
                  </span>
                )}
                {c.score !== undefined && (
                  <span
                    style={{
                      marginLeft: 4,
                      fontSize: 9,
                      fontWeight: 600,
                      color: confidenceColor(c.score),
                      opacity: 0.9,
                    }}
                  >
                    {confidenceLabel(c.score)}
                  </span>
                )}
              </span>
            ))}
            {retrievalScore !== undefined && (
              <span
                className="citation-chip"
                style={{
                  marginLeft: 'auto',
                  color: confidenceColor(retrievalScore),
                  borderColor: confidenceColor(retrievalScore) + '44',
                  background: confidenceColor(retrievalScore) + '18',
                }}
                title="Best retrieval similarity score"
              >
                {Math.round(retrievalScore * 100)}% match
              </span>
            )}
          </div>
        )}

        {/* Out-of-scope badge (no citations) */}
        {isRefusal && !hasCitations && (
          <div className="refusal-badge">
            <ShieldOff size={10} />
            Out of scope — not found in PDF
          </div>
        )}
      </div>
    </div>
  );
}
