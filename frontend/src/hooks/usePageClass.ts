import { useEffect } from 'react';

/**
 * Scope the page's CSS by putting a class on <body>.
 *
 * styles.css keeps the two pages' rules under `.page-home` and `.page-editor`
 * because their `button` baselines genuinely disagree -- the editor's is a
 * full-width solid navy, the filesystem's is a bare element styled by `.btn`.
 * Neither can be hoisted to the root without breaking the other.
 */
export function usePageClass(cls: 'page-home' | 'page-editor') {
  useEffect(() => {
    document.body.classList.add(cls);
    return () => document.body.classList.remove(cls);
  }, [cls]);
}
