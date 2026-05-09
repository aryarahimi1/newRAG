<script lang="ts">
	import HamburgerIcon from '$lib/components/icons/HamburgerIcon.svelte';
	import ThemeGlyph from '$lib/components/icons/ThemeGlyph.svelte';
	import type { Theme } from '$lib/theme';

	let {
		isMobile,
		theme,
		hasMessages,
		hasPipelineResult,
		railCollapsed,
		onOpenRail,
		onToggleTheme,
		onClearChat
	}: {
		isMobile: boolean;
		theme: Theme;
		hasMessages: boolean;
		hasPipelineResult: boolean;
		railCollapsed: boolean;
		onOpenRail: () => void;
		onToggleTheme: () => void;
		onClearChat: () => void;
	} = $props();
</script>

<header class="topbar">
	<div class="topbar-primary">
		{#if isMobile}
			<button
				type="button"
				class="icon-button mobile-menu-btn"
				onclick={onOpenRail}
				aria-label="Open chat history"
				aria-controls="session-rail-panel"
				aria-expanded={!railCollapsed}
			>
				<HamburgerIcon />
			</button>
		{/if}
		<div class="topbar-titles">
			<p class="eyebrow">Citation-backed reference</p>
			<h2>Ask about drug interactions</h2>
			<p class="subtle">PII is redacted before retrieval. Answers cite FDA and NIH public sources.</p>
		</div>
	</div>

	<div class="topbar-actions">
		<button
			type="button"
			class="ghost small theme-toggle"
			onclick={onToggleTheme}
			aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
			title={theme === 'light' ? 'Dark mode' : 'Light mode'}
		>
			<ThemeGlyph {theme} />
		</button>
		<button type="button" class="ghost small" onclick={onClearChat} disabled={!hasMessages && !hasPipelineResult}>
			Clear
		</button>
	</div>
</header>

<style>
	.topbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
	}

	.topbar-primary {
		display: flex;
		align-items: flex-start;
		gap: 0.65rem;
		min-width: 0;
		flex: 1;
	}

	.topbar-titles {
		min-width: 0;
	}

	.mobile-menu-btn {
		flex: 0 0 auto;
		margin-top: 0.1rem;
	}

	.mobile-menu-btn :global(.hamburger-icon) {
		display: block;
		width: 1.2rem;
		height: 1.2rem;
	}

	.eyebrow {
		margin: 0;
		color: var(--eyebrow);
		text-transform: uppercase;
		letter-spacing: 0.12em;
		font-size: 0.72rem;
		font-weight: 500;
	}

	h2 {
		margin: 0;
		color: var(--text-heading);
		font-size: clamp(1.45rem, 2vw, 2.2rem);
	}

	.subtle {
		margin: 0.25rem 0 0;
		color: var(--text-muted);
	}

	.topbar-actions {
		display: flex;
		align-items: center;
		gap: 0.65rem;
		flex-wrap: wrap;
		justify-content: flex-end;
	}

	button {
		font: inherit;
		cursor: pointer;
		border: 0;
	}

	button:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	.ghost,
	.icon-button {
		border-radius: 999px;
		transition:
			transform 0.16s ease,
			border-color 0.16s ease,
			background 0.16s ease;
	}

	.ghost {
		border: 1px solid var(--btn-ghost-border);
		background: var(--btn-ghost-bg);
		color: var(--btn-ghost-fg);
		font-weight: 500;
		padding: 0.62rem 0.85rem;
	}

	.ghost:hover:not(:disabled),
	.icon-button:hover {
		transform: translateY(-1px);
	}

	.theme-toggle {
		display: inline-grid;
		place-items: center;
		min-width: 2.45rem;
		padding-left: 0.62rem;
		padding-right: 0.62rem;
	}

	.theme-toggle :global(.theme-icon) {
		width: 1.15rem;
		height: 1.15rem;
		display: block;
	}

	.icon-button {
		width: 2.45rem;
		height: 2.45rem;
		border: 1px solid var(--btn-ghost-border);
		background: var(--btn-ghost-bg);
		color: var(--btn-ghost-fg);
		font-weight: 500;
		display: inline-grid;
		place-items: center;
		padding: 0;
	}

	.small {
		padding: 0.62rem 0.85rem;
	}

	@media (max-width: 920px) {
		.topbar {
			align-items: flex-start;
			flex-direction: column;
		}
	}
</style>
