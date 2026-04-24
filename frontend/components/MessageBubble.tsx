'use client';

import { BookOpen, FileText } from 'lucide-react';
import { Citation } from '@/lib/types';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Citation[];
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

export default function MessageBubble({
  role,
  content,
  sources,
  timestamp,
  isError,
}: MessageBubbleProps) {
  const hasCitations = sources && sources.length > 0;

  return (
    <div className={`message-wrapper ${role}`}>
      {/* Avatar + time row */}
      <div
        className="message-meta"
        style={{ flexDirection: role === 'user' ? 'row-reverse' : 'row' }}
      >
        <div className={`message-avatar ${role}`}>
          {role === 'user' ? 'U' : <FileText size={14} />}
        </div>
        <span className="message-time">{formatTime(timestamp)}</span>
      </div>

      {/* Bubble + (optional) citation block */}
      <div style={{ maxWidth: '72%', display: 'flex', flexDirection: 'column' }}>
        <div
          className={`message-bubble ${role} ${isError ? 'error-bubble' : ''}`}
          style={{
            borderRadius:
              role === 'user'
                ? '16px 16px 4px 16px'
                : hasCitations
                ? '16px 16px 0 4px'
                : '16px 16px 16px 4px',
          }}
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
              <span key={idx} className="citation-chip">
                <FileText size={9} />
                p.{c.page_number}
                {sources.some(
                  (s, i) => i !== idx && s.pdf_name !== c.pdf_name,
                ) && (
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
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
