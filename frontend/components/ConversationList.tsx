'use client';

import { useState } from 'react';
import { MessageSquare, Plus, Pencil, Trash2, Check, X } from 'lucide-react';
import { ConversationThread } from '@/lib/types';

interface Props {
  conversations: ConversationThread[];
  activeSessionId: string;
  onNew: () => void;
  onSwitch: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const m = Math.floor(diff / 60000);
  const h = Math.floor(diff / 3600000);
  const d = Math.floor(diff / 86400000);
  if (m < 1) return 'Just now';
  if (m < 60) return `${m}m ago`;
  if (h < 24) return `${h}h ago`;
  if (d === 1) return 'Yesterday';
  if (d < 7) return `${d}d ago`;
  return new Date(ts).toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function ConversationList({ conversations, activeSessionId, onNew, onSwitch, onRename, onDelete }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const startEdit = (conv: ConversationThread, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const commitEdit = () => {
    if (editingId && editTitle.trim()) onRename(editingId, editTitle.trim());
    setEditingId(null);
  };

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this conversation? This cannot be undone.')) return;
    onDelete(id);
  };

  return (
    <div className="conv-wrap">
      <button className="btn-new-conv" onClick={onNew} id="new-conversation-btn">
        <Plus size={14} />
        New conversation
      </button>

      <div className="conv-list">
        {conversations.length === 0 ? (
          <div className="conv-empty">
            <MessageSquare size={28} className="conv-empty-icon" />
            <p>No conversations yet</p>
          </div>
        ) : (
          conversations
            .slice()
            .sort((a, b) => b.updatedAt - a.updatedAt)
            .map((conv) => {
              const isActive = conv.id === activeSessionId;
              const isEditing = editingId === conv.id;

              return (
                <div
                  key={conv.id}
                  className={`conv-item ${isActive ? 'active' : ''}`}
                  onClick={() => !isEditing && onSwitch(conv.id)}
                  id={`conv-${conv.id}`}
                >
                  <div className="conv-item-icon">
                    <MessageSquare size={13} />
                  </div>

                  <div className="conv-item-body">
                    {isEditing ? (
                      <input
                        className="conv-rename-input"
                        value={editTitle}
                        autoFocus
                        onChange={(e) => setEditTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') commitEdit();
                          if (e.key === 'Escape') setEditingId(null);
                        }}
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <span className="conv-title">{conv.title}</span>
                    )}
                    <span className="conv-meta">
                      {relativeTime(conv.updatedAt)}
                      {conv.previewText && (
                        <span className="conv-preview"> · {conv.previewText}</span>
                      )}
                    </span>
                  </div>

                  <div className="conv-actions" onClick={(e) => e.stopPropagation()}>
                    {isEditing ? (
                      <>
                        <button className="conv-action-btn" onClick={commitEdit} title="Save">
                          <Check size={12} />
                        </button>
                        <button className="conv-action-btn" onClick={() => setEditingId(null)} title="Cancel">
                          <X size={12} />
                        </button>
                      </>
                    ) : (
                      <>
                        <button className="conv-action-btn" onClick={(e) => startEdit(conv, e)} title="Rename">
                          <Pencil size={12} />
                        </button>
                        <button className="conv-action-btn delete" onClick={(e) => handleDelete(conv.id, e)} title="Delete">
                          <Trash2 size={12} />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              );
            })
        )}
      </div>
    </div>
  );
}
