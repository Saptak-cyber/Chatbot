'use client';

import { useEffect, useState } from 'react';
import { checkHealth } from '@/lib/api';

export default function ColdStartBanner() {
  const [isChecking, setIsChecking] = useState(true);
  const [isColdStart, setIsColdStart] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [dots, setDots] = useState('');

  useEffect(() => {
    let mounted = true;
    let timeoutId: NodeJS.Timeout;

    const checkBackend = async () => {
      if (!mounted) return;

      const isHealthy = await checkHealth();

      if (!mounted) return;

      if (isHealthy) {
        setIsChecking(false);
        setIsColdStart(false);
      } else {
        setIsColdStart(true);
        setRetryCount((prev) => prev + 1);
        
        // Retry every 5 seconds
        timeoutId = setTimeout(checkBackend, 5000);
      }
    };

    checkBackend();

    return () => {
      mounted = false;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, []);

  // Animated dots effect
  useEffect(() => {
    if (!isColdStart) return;

    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
    }, 500);

    return () => clearInterval(interval);
  }, [isColdStart]);

  if (!isColdStart && !isChecking) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
        backgroundColor: '#ff9800',
        color: '#000',
        padding: '16px',
        textAlign: 'center',
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
        <div
          style={{
            width: '20px',
            height: '20px',
            border: '3px solid #000',
            borderTopColor: 'transparent',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
          }}
        />
        <div>
          <strong>Backend is waking up{dots}</strong>
          <div style={{ fontSize: '14px', marginTop: '4px', opacity: 0.8 }}>
            Render.com free tier cold start in progress. This usually takes 30-60 seconds.
            {retryCount > 0 && ` (Attempt ${retryCount})`}
          </div>
        </div>
      </div>
      <style jsx>{`
        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
}
