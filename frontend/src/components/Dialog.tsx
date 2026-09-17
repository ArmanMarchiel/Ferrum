import { useEffect, useRef, type ReactNode } from 'react';

interface DialogProps {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
  /** The action row. Rendered inside `.dlgact`. */
  actions?: ReactNode;
  error?: string;
}

/**
 * The modal shell both pages use: save, rename, delete, export.
 *
 * Escape closes and the first field takes focus on open -- the original pages
 * wired both by hand per dialog, and the rename dialog was the only one that
 * actually got the focus call, so the others opened needing a click.
 */
export function Dialog({
  open,
  title,
  onClose,
  children,
  actions,
  error,
}: DialogProps) {
  const body = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    // Focus the first field, so the dialog is typeable without a click.
    const field = body.current?.querySelector<HTMLElement>(
      'input, select, textarea',
    );
    field?.focus();
    if (field instanceof HTMLInputElement) field.select();
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="veil"
      onMouseDown={(e) => {
        // Only a click on the backdrop itself dismisses; a drag that ends
        // outside the dialog should not.
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="dlg" role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        <div ref={body}>{children}</div>
        {error ? <div className="err">{error}</div> : null}
        {actions ? <div className="dlgact">{actions}</div> : null}
      </div>
    </div>
  );
}
