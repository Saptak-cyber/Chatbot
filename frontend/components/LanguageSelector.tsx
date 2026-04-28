'use client';

import { useState, useRef, useEffect } from 'react';
import { Globe, ChevronDown, Check } from 'lucide-react';

/** Languages officially supported by Llama 3.1 8B Instant (Meta's documented list). */
export const SUPPORTED_LANGUAGES = [
  { code: 'auto', flag: '🌐', native: 'Auto-detect', english: 'Detect from query' },
  { code: 'en',   flag: '🇬🇧', native: 'English',    english: 'English'    },
  { code: 'de',   flag: '🇩🇪', native: 'Deutsch',    english: 'German'     },
  { code: 'fr',   flag: '🇫🇷', native: 'Français',   english: 'French'     },
  { code: 'it',   flag: '🇮🇹', native: 'Italiano',   english: 'Italian'    },
  { code: 'pt',   flag: '🇵🇹', native: 'Português',  english: 'Portuguese' },
  { code: 'hi',   flag: '🇮🇳', native: 'हिंदी',       english: 'Hindi'      },
  { code: 'es',   flag: '🇪🇸', native: 'Español',    english: 'Spanish'    },
  { code: 'th',   flag: '🇹🇭', native: 'ภาษาไทย',    english: 'Thai'       },
] as const;

export type LanguageCode = typeof SUPPORTED_LANGUAGES[number]['code'];

interface LanguageSelectorProps {
  selectedLanguage: string;
  onLanguageChange: (code: string) => void;
}

export default function LanguageSelector({
  selectedLanguage,
  onLanguageChange,
}: LanguageSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const current =
    SUPPORTED_LANGUAGES.find((l) => l.code === selectedLanguage) ??
    SUPPORTED_LANGUAGES[0];

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [isOpen]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [isOpen]);

  return (
    <div className="lang-selector" ref={ref}>
      <button
        className={`lang-trigger ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen((o) => !o)}
        title="Select response language — 8 languages supported by Llama 3.1 8B"
        id="language-selector-btn"
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span className="lang-flag">{current.flag}</span>
        <span className="lang-label">
          {current.code === 'auto' ? 'Auto' : current.native}
        </span>
        <ChevronDown size={11} className={`lang-chevron ${isOpen ? 'rotated' : ''}`} />
      </button>

      {isOpen && (
        <div className="lang-dropdown" role="listbox">
          <div className="lang-dropdown-header">
            <Globe size={10} />
            <span>Llama 3.1 8B · 8 languages</span>
          </div>

          {SUPPORTED_LANGUAGES.map((lang) => {
            const isActive = selectedLanguage === lang.code;
            return (
              <button
                key={lang.code}
                role="option"
                aria-selected={isActive}
                className={`lang-option ${isActive ? 'active' : ''}`}
                onClick={() => {
                  onLanguageChange(lang.code);
                  setIsOpen(false);
                }}
              >
                <span className="lang-option-flag">{lang.flag}</span>
                <span className="lang-option-names">
                  <span className="lang-option-native">{lang.native}</span>
                  <span className="lang-option-english">{lang.english}</span>
                </span>
                {isActive && (
                  <Check size={11} className="lang-option-check" />
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
