<script lang="ts">
	import { browser } from '$app/environment';
	import { tick } from 'svelte';

	import PharmaMark from '$lib/components/icons/PharmaMark.svelte';
	import RailToggleIcon from '$lib/components/icons/RailToggleIcon.svelte';
	import { chatListPreview, chatListTitle, formatTime } from '$lib/chat-format';
	import type { ChatSession } from '$lib/types/chat';

	let {
		sessions,
		activeChatId,
		railCollapsed,
		isMobile,
		railHeadingId,
		onSelectChat,
		onNewChat,
		onToggleRail
	}: {
		sessions: ChatSession[];
		activeChatId: string;
		railCollapsed: boolean;
		isMobile: boolean;
		railHeadingId: string;
		onSelectChat: (id: string) => void;
		onNewChat: () => void;
		onToggleRail: () => void;
	} = $props();

	let railDialogOpen = $derived(isMobile && !railCollapsed);

	let panelEl = $state<HTMLElement | undefined>(undefined);

	$effect(() => {
		if (!browser || !railDialogOpen || !panelEl) return;
		const prevFocus = document.activeElement;
		void tick().then(() => {
			panelEl?.querySelector<HTMLElement>('[data-rail-initial-focus]')?.focus();
		});
		return () => {
			if (prevFocus instanceof HTMLElement && document.contains(prevFocus)) {
				prevFocus.focus();
			}
		};
	});

	let rows = $derived(
		sessions.map((session) => ({
			id: session.id,
			title: chatListTitle(session),
			preview: chatListPreview(session),
			timeLabel: formatTime(session.updatedAt),
			active: session.id === activeChatId
		}))
	);
</script>

<aside
	id="session-rail-panel"
	class="session-rail"
	aria-label="Current session chats"
	role={railDialogOpen ? 'dialog' : undefined}
	aria-modal={railDialogOpen ? true : undefined}
	aria-labelledby={railDialogOpen ? railHeadingId : undefined}
	{@attach (el) => {
		panelEl = el;
		return () => {
			panelEl = undefined;
		};
	}}
>
	<div class="rail-header">
		<div class="brand-mark" aria-hidden="true"><PharmaMark /></div>
		{#if !railCollapsed}
			<div>
				<p class="eyebrow">Medication Reference</p>
				<h1 id={railHeadingId}>Session</h1>
			</div>
		{/if}
	</div>

	<div class="rail-actions">
		<button
			type="button"
			class="primary small"
			onclick={onNewChat}
			aria-label="Start a new chat"
			data-rail-initial-focus={railDialogOpen || undefined}
		>
			<span aria-hidden="true">+</span>
			{#if !railCollapsed}<span>New chat</span>{/if}
		</button>
		<button
			type="button"
			class="icon-button rail-toggle"
			onclick={onToggleRail}
			aria-label={railCollapsed ? 'Expand chat history' : 'Collapse chat history'}
			aria-expanded={!railCollapsed}
		>
			<RailToggleIcon />
		</button>
	</div>

	{#if !railCollapsed}
		<nav class="chat-list" aria-label="Chats in this session">
			{#each rows as row (row.id)}
				<button
					type="button"
					class={['chat-tab', row.active && 'active'].filter(Boolean).join(' ')}
					onclick={() => onSelectChat(row.id)}
					aria-current={row.active ? 'page' : undefined}
				>
					<span class="chat-title">{row.title}</span>
					<span class="chat-preview">{row.preview}</span>
					<span class="chat-time">{row.timeLabel}</span>
				</button>
			{/each}
		</nav>
	{/if}
</aside>

<style>
	.session-rail {
		border-right: 1px solid var(--rail-border);
		background: var(--rail-bg);
		padding: 1rem;
		display: flex;
		flex-direction: column;
		gap: 1rem;
		min-width: 0;
	}

	.rail-header,
	.rail-actions {
		display: flex;
		align-items: center;
	}

	.rail-header {
		gap: 0.75rem;
		min-height: 3rem;
	}

	.brand-mark {
		display: grid;
		place-items: center;
		width: 2.75rem;
		height: 2.75rem;
		flex: 0 0 auto;
		border-radius: 10px;
		background: var(--brand-mark-bg);
		border: 1px solid var(--brand-mark-border);
		color: var(--brand-mark-fg);
		padding: 0.45rem;
		box-sizing: border-box;
	}

	.brand-mark :global(svg) {
		width: 100%;
		height: 100%;
		display: block;
	}

	.eyebrow {
		margin: 0;
		color: var(--eyebrow);
		text-transform: uppercase;
		letter-spacing: 0.12em;
		font-size: 0.72rem;
		font-weight: 500;
	}

	h1,
	p {
		margin-top: 0;
	}

	.rail-header h1 {
		margin: 0;
		color: var(--text-heading);
	}

	.rail-actions {
		gap: 0.5rem;
	}

	button {
		border: 0;
		cursor: pointer;
		font: inherit;
	}

	button:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	.primary,
	.icon-button {
		border-radius: 999px;
		transition:
			transform 0.16s ease,
			border-color 0.16s ease,
			background 0.16s ease;
	}

	.primary {
		background: var(--btn-primary);
		color: var(--brand-mark-fg);
		font-weight: 600;
		padding: 0.8rem 1.05rem;
	}

	.primary:hover:not(:disabled) {
		background: var(--btn-primary-hover);
	}

	.primary:hover:not(:disabled),
	.icon-button:hover,
	.chat-tab:hover {
		transform: translateY(-1px);
	}

	.small {
		padding: 0.62rem 0.85rem;
	}

	.rail-actions .primary {
		flex: 1;
		display: inline-flex;
		justify-content: center;
		gap: 0.4rem;
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

	.rail-toggle :global(.rail-toggle-svg) {
		width: 1.15rem;
		height: 1.15rem;
		color: var(--btn-ghost-fg);
	}

	/* Collapsed: chevron right (open rail); expanded: chevron left (minimize) */
	.rail-toggle :global(.rail-toggle-path) {
		transform-origin: 12px 12px;
		transform: scaleX(-1);
	}

	:global(.rail-minimized) .rail-toggle :global(.rail-toggle-path) {
		transform: none;
	}

	.chat-list {
		display: flex;
		flex-direction: column;
		gap: 0.55rem;
		overflow: auto;
		padding-right: 0.15rem;
	}

	.chat-tab {
		text-align: left;
		display: grid;
		gap: 0.25rem;
		padding: 0.8rem;
		border-radius: var(--radius-md);
		transition:
			transform 0.16s ease,
			border-color 0.16s ease,
			background 0.16s ease;
		color: var(--tab-fg);
		background: var(--tab-bg);
		border: 1px solid var(--border);
	}

	.chat-tab.active {
		background: var(--tab-active-bg);
		border-color: var(--tab-active-border);
	}

	.chat-title,
	.chat-preview {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.chat-title {
		font-weight: 600;
	}

	.chat-preview,
	.chat-time {
		color: var(--text-muted);
		font-size: 0.8rem;
	}

	@media (max-width: 920px) {
		.session-rail {
			position: fixed;
			z-index: 40;
			top: 0;
			bottom: 0;
			left: 0;
			width: min(20rem, 90vw);
			max-width: 100%;
			height: 100dvh;
			max-height: 100dvh;
			margin: 0;
			border-right: 1px solid var(--rail-border);
			border-top: none;
			box-shadow: none;
			transform: translate3d(-100%, 0, 0);
			transition: transform 0.22s ease, box-shadow 0.22s ease;
			overflow-y: auto;
			overflow-x: hidden;
			overscroll-behavior: contain;
			-webkit-overflow-scrolling: touch;
			padding: 1rem;
			padding-top: max(1rem, env(safe-area-inset-top));
			padding-left: max(1rem, env(safe-area-inset-left));
			padding-bottom: max(1rem, env(safe-area-inset-bottom));
		}

		:global(.app-shell:not(.rail-minimized)) .session-rail {
			transform: translate3d(0, 0, 0);
			box-shadow: 4px 0 32px color-mix(in srgb, var(--shell-fg) 16%, transparent);
		}

		@media (prefers-reduced-motion: reduce) {
			.session-rail {
				transition: none;
			}
		}
	}
</style>
