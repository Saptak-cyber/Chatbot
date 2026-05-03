'use client';

import { useState } from 'react';
import { BookOpen, FileText, ShieldOff, Copy, Check } from 'lucide-react';
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

function isTableRow(line: string): boolean {
  return line.trim().startsWith('|') && line.trim().endsWith('|');
}

function isSeparatorRow(line: string): boolean {
  return isTableRow(line) && /^[\s|:|-]+$/.test(line);
}

function parseTableCells(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

/**
 * Normalise LLM output before line-by-line rendering.
 *
 * The LLM sometimes emits inline asterisk bullets without newlines:
 *   "…policy [Page 9]. * Sick leave…"
 * This step splits those into proper separate lines so the renderer
 * picks them up as list items.
 *
 * It also removes stray lines that are only "-", "*", or "•" (broken markdown)
 * and merges the following line as a proper "- ..." bullet when needed.
 */
function preprocessContent(text: string): string {
  const rawLines = text.split('\n');
  const fixedLines: string[] = [];

  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i];
    const trimmed = line.trim();

    // Drop orphan bullet markers (line is only * / - / •) — common LLM glitch.
    // Look ahead PAST any blank lines to find the real content line.
    if (/^[-•*]\s*$/.test(trimmed)) {
      // Advance j past blank lines
      let j = i + 1;
      while (j < rawLines.length && !rawLines[j].trim()) j++;

      if (j < rawLines.length) {
        const nt = rawLines[j].trim();
        if (
          nt &&
          !/^[-•*]\s/.test(nt) &&
          !/^\d+\.\s/.test(nt) &&
          !nt.startsWith('|')
        ) {
          // Absorb the orphan marker + all blank lines + content into one bullet
          fixedLines.push('- ' + rawLines[j].replace(/^\s+/, ''));
          i = j; // skip to the merged line
          continue;
        }
      }
      // No valid next line — just drop the orphan marker
      continue;
    }

    fixedLines.push(line);
  }

  return fixedLines
    .join('\n')
    // "text. * Item" → "text.\n* Item"  (inline bullet after sentence)
    .replace(/\.\s*\*\s+(?=[^\s])/g, '.\n* ')
    // "[Citation]. * Item" → "[Citation].\n* Item"
    .replace(/(\[[^\]]+\])\s*\.\s*\*\s+/g, '$1.\n* ')
    // "[Citation] * Item" → "[Citation]\n* Item"  (no period between bracket and bullet)
    .replace(/(\[[^\]]+\])\s*\*\s+(?=[^\s*])/g, '$1\n* ')
    // ": * Item" → ":\n* Item"  (inline bullet after colon — common in history-based responses)
    .replace(/:\s*\*\s+(?=[^\s*])/g, ':\n* ')
    // "text. - Item" (dash variant, only if followed by capital to avoid hyphens)
    .replace(/\.\s*-\s+(?=[A-Z])/g, '.\n- ')
    // ": - Item" → ":\n- Item"  (dash variant after colon)
    .replace(/:\s*-\s+(?=[A-Z])/g, ':\n- ')
    // Ensure "**Label:** rest of line" is on its own line — but only when it's
    // NOT already a list-item start (- / • / *). Excluding those characters from
    // the look-behind prevents breaking "- **Label**: content" into a stray dash.
    // Also restrict [^\n*]+ to avoid greedy cross-line matches.
    .replace(/([^\n\-•*])\s*(\*\*[^\n*]+\*\*:)/g, '$1\n\n$2');
}

function renderContent(text: string) {
  // Normalise inline bullets and bold-label patterns before line-splitting
  const normalised = preprocessContent(text);
  // Enhanced markdown-like rendering with bold, lists, tables
  const lines = normalised.split('\n');
  const elements: React.ReactNode[] = [];
  let inList = false;
  let listItems: React.ReactNode[] = [];
  // Table accumulation
  let tableLines: string[] = [];
  let inTable = false;

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

  const flushTable = (index: number) => {
    if (!inTable || tableLines.length === 0) return;
    inTable = false;

    // Must have at least a header + separator row
    if (tableLines.length < 2) {
      // Render as plain text if not a valid table
      tableLines.forEach((tl, ti) =>
        elements.push(<p key={`tp-${index}-${ti}`}>{tl}</p>)
      );
      tableLines = [];
      return;
    }

    const headerCells = parseTableCells(tableLines[0]);
    // tableLines[1] is separator — skip it
    const dataRows = tableLines.slice(2);

    elements.push(
      <div key={`table-wrap-${index}`} className="md-table-wrap">
        <table className="md-table">
          <thead>
            <tr>
              {headerCells.map((cell, ci) => (
                <th key={ci}>{processInline(cell)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataRows
              .filter((row) => isTableRow(row))
              .map((row, ri) => (
                <tr key={ri}>
                  {parseTableCells(row).map((cell, ci) => (
                    <td key={ci}>{processInline(cell)}</td>
                  ))}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    );
    tableLines = [];
  };

  // Process inline formatting (bold, bold-label, plain)
  const processInline = (text: string): React.ReactNode => {
    const parts: React.ReactNode[] = [];
    // Regex: **label:** (bold ending with colon — sub-heading pattern) OR **bold** (normal bold)
    const inlineRegex = /(\*\*([^*]+?):\*\*|\*\*([^*]+?)\*\*)/g;
    let lastIndex = 0;
    let key = 0;
    let match;

    while ((match = inlineRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      if (match[2]) {
        // **label:** — render as accent-coloured label
        parts.push(
          <span
            key={`label-${key++}`}
            style={{ color: 'var(--accent-bright)', fontWeight: 600 }}
          >
            {match[2]}:
          </span>
        );
      } else {
        // **bold** — render as bold
        parts.push(<strong key={`bold-${key++}`}>{match[3]}</strong>);
      }
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    return parts.length > 0 ? <>{parts}</> : text;
  };

  lines.forEach((line, i) => {
    const trimmed = line.trim();

    // ── Table detection ───────────────────────────────────────────────────
    if (isTableRow(trimmed)) {
      // flush list before starting a table
      flushList(i);
      inTable = true;
      tableLines.push(trimmed);
      return;
    }

    // Line is NOT a table row — flush any accumulated table
    if (inTable) {
      flushTable(i);
    }

    // Empty line - add spacing
    if (!trimmed) {
      flushList(i);
      elements.push(<div key={`space-${i}`} style={{ height: 8 }} />);
      return;
    }

    // List item: "- ", "• ", "* ", "*Item", numbered
    const isListLine =
      trimmed.match(/^[-•]\s/) ||
      trimmed.match(/^\*\s/) ||
      trimmed.match(/^\*(?!\*)\S/) ||
      trimmed.match(/^\d+\.\s/);
    if (isListLine) {
      inList = true;
      let content = trimmed;
      if (/^\d+\.\s/.test(content)) content = content.replace(/^\d+\.\s/, '');
      else if (/^[-•]\s/.test(content)) content = content.replace(/^[-•]\s/, '');
      else if (/^\*\s/.test(content)) content = content.replace(/^\*\s+/, '');
      else if (/^\*(?!\*)/.test(content)) content = content.replace(/^\*/, '');
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
    const isHeading =
      trimmed.startsWith('#') ||
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
            letterSpacing: '-0.2px',
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
        style={{ marginBottom: 10, lineHeight: 1.7, color: 'var(--text-primary)' }}
      >
        {processInline(trimmed)}
      </p>
    );
  });

  // Flush any remaining list or table
  flushList(lines.length);
  flushTable(lines.length);

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
  const [copied, setCopied] = useState(false);

  const hasCitations = sources && sources.length > 0;
  // isGrounded is undefined for user messages — treat undefined as true
  const isRefusal = role === 'assistant' && isGrounded === false;
  const isMultiPdf = hasCitations
    ? new Set(sources.map((s) => s.pdf_name)).size > 1
    : false;

  const handleCopy = () => {
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

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
          className={`message-bubble ${role} ${isError ? 'error-bubble' : ''} ${isRefusal ? 'refusal-bubble' : ''} ${hasCitations ? 'has-citations' : ''} ${role === 'assistant' ? 'bubble-copyable' : ''}`}
        >
          {renderContent(content)}

          {/* Copy button — assistant messages only, shown on bubble hover */}
          {role === 'assistant' && content && (
            <button
              className={`btn-copy-msg ${copied ? 'copied' : ''}`}
              onClick={handleCopy}
              title={copied ? 'Copied!' : 'Copy response'}
              aria-label="Copy message"
            >
              {copied ? <Check size={11} /> : <Copy size={11} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          )}
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
