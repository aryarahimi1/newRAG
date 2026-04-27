export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'medication-reference-theme';

export function getStoredTheme(): Theme | null {
	if (typeof localStorage === 'undefined') return null;
	try {
		const v = localStorage.getItem(STORAGE_KEY);
		if (v === 'light' || v === 'dark') return v;
	} catch {
		/* ignore */
	}
	return null;
}

export function getSystemTheme(): Theme {
	if (typeof window === 'undefined') return 'light';
	return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function applyTheme(theme: Theme): void {
	if (typeof document === 'undefined') return;
	document.documentElement.dataset.theme = theme;
	try {
		localStorage.setItem(STORAGE_KEY, theme);
	} catch {
		/* quota / private mode */
	}
}

export function resolveInitialTheme(): Theme {
	return getStoredTheme() ?? getSystemTheme();
}
