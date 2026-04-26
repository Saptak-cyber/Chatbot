'use client';

import { BookOpen, FileText, ShieldOff } from 'lucide-react';
import { Citation } from '@/lib/types';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Citation[];
  isGrounded?: boolean;
  retrievalScore?: number;
  confidenceLevel?: 'high' | 'medium' | 'low';
  numSources?: number;
  timestamp: Date;
  isError?: boolean;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function renderContent(text: string) {
  // Enhanced markdown-like rendering with better formatting
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let inList = false;
  let listItems: React.ReactNode[] = [];

  const flushList = (index: number) => {
    if (inList && listItems.length > 0) {
      elements.push(
        <ul key={`list-${index}`} style={{ marginBottom: 12, paddingLeft: 20 }}>
          {listItems}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  };

  lines.forEach((line, i) => {
    const trimmed = line.trim();

    // Empty line - add spacing
    if (!trimmed) {
      flushList(i);
      elements.push(<div key={`space-${i}`} style={{ height: 8 }} />);
      return;
    }

    // Process inline formatting
    const processInline = (text: string) => {
      const parts: React.ReactNode[] = [];
      let remaining = text;
      let key = 0;

      // Bold: **text**
      const boldRegex = /\*\*([^*]+)\*\*/g;
      let lastIndex = 0;
      let match;

      while ((match = boldRegex.exec(text)) !== null) {
        if (match.index > lastIndex) {
          parts.push(text.substring(lastIndex, match.index));
        }
        parts.push(<strong key={`bold-${key++}`}>{match[1]}</strong>);
        lastIndex = match.index + match[0].length;
      }

      if (lastIndex < text.length) {
        parts.push(text.substring(lastIndex));
      }

      return parts.length > 0 ? parts : text;
    };

    // List item: starts with -, •, *, or number.
    if (trimmed.match(/^[-•*]\s/) || trimmed.match(/^\d+\.\s/)) {
      inList = true;
      const content = trimmed.replace(/^[-•*]\s/, '').replace(/^\d+\.\s/, '');
      listItems.push(
        <li key={`li-${i}`} style={{ marginBottom: 6, lineHeight: 1.6 }}>
          {processInline(content)}
        </li>
      );
      return;
    }

    // Not a list item, flush any pending list
    flushList(i);

    // Heading: starts with # or is all caps and short
    const isHeading = trimmed.startsWith('#') || 
                     (trimmed === trimmed.toUpperCase() && 
                      trimmed.length < 60 && 
                      trimmed.length > 3 &&
                      !trimmed.includes('[Page'));

    if (isHeading) {
      const headingText = trimmed.replace(/^#+\s*/, '');
      elements.push(
        <h3 
          key={`heading-${i}`} 
          style={{ 
            fontWeight: 600, 
            fontSize: '15px',
            marginBottom: 8,
            marginTop: 12,
            color: 'var(--text-primary)',
            letterSpacing: '-0.2px'
          }}
        >
          {processInline(headingText)}
        </h3>
      );
      return;
    }

    // Regular paragraph
    elements.push(
      <p 
        key={`p-${i}`} 
        style={{ 
          marginBottom: 10, 
          lineHeight: 1.7,
          color: 'var(--text-primary)'
        }}
      >
        {processInline(trimmed)}
      </p>
    );
  });

  // Flush any remaining list
  flushList(lines.length);

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
  confidenceLevel,
  numSources,
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
              <span
                key={idx}
                className="citation-chip"
                title={[
                  c.pdf_name,
                  c.section ? `§ ${c.section}` : null,
                  `Page ${c.page_number}`,
                  c.score !== undefined ? `${Math.round(c.score * 100)}% match` : null,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              >
                <FileText size={9} />
                p.{c.page_number}
                {c.section && (
                  <span
                    style={{
                      opacity: 0.75,
                      maxWidth: 120,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontStyle: 'italic',
                    }}
                  >
                    &nbsp;§&nbsp;{c.section}
                  </span>
                )}
                {!c.section && isMultiPdf && (
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
            {/* Confidence level badge */}
            {confidenceLevel && (
              <span
                className="citation-chip"
                style={{
                  marginLeft: 'auto',
                  color: confidenceLevel === 'high' ? 'var(--success)' : confidenceLevel === 'medium' ? 'var(--warning)' : '#f87171',
                  borderColor: (confidenceLevel === 'high' ? 'var(--success)' : confidenceLevel === 'medium' ? 'var(--warning)' : '#f87171') + '44',
                  background: (confidenceLevel === 'high' ? 'var(--success)' : confidenceLevel === 'medium' ? 'var(--warning)' : '#f87171') + '18',
                  fontWeight: 600,
                }}
                title={`Confidence: ${confidenceLevel} (based on retrieval score ${retrievalScore ? Math.round(retrievalScore * 100) + '%' : 'N/A'})`}
              >
                {confidenceLevel.toUpperCase()} confidence
              </span>
            )}
            {/* Fallback to retrieval score if no confidence level */}
            {!confidenceLevel && retrievalScore !== undefined && (
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
            <span className="refusal-badge-icon">
              <ShieldOff size={11} />
            </span>
            <span className="refusal-badge-dot" />
            <span>Out of scope</span>
            <span className="refusal-badge-sep">·</span>
            <span className="refusal-badge-sub">No relevant content found in PDF</span>
          </div>
        )}
      </div>
    </div>
  );
}
