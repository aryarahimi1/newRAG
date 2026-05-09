<script lang="ts">
	import ChatMessage from '$lib/components/ChatMessage.svelte';
	import EmptyState from '$lib/components/EmptyState.svelte';
	import AssistantAvatarGlyph from '$lib/components/icons/AssistantAvatarGlyph.svelte';
	import type { ChatTurn } from '$lib/types/chat';

	let {
		hasMessages,
		currentMessages,
		loading,
		streamingMsgId,
		streamingPreAnswer,
		lastSentUserId,
		latestStatus,
		sampleQuestions,
		threadAttach,
		onAskSample
	}: {
		hasMessages: boolean;
		currentMessages: ChatTurn[];
		loading: boolean;
		streamingMsgId: string | null;
		streamingPreAnswer: boolean;
		lastSentUserId: string | null;
		latestStatus: string;
		sampleQuestions: string[];
		threadAttach: (node: HTMLDivElement) => void | (() => void);
		onAskSample: (q: string) => void;
	} = $props();
</script>

<div class="thread" {@attach threadAttach}>
	{#if !hasMessages}
		<EmptyState {sampleQuestions} {loading} onAskSample={onAskSample} />
	{:else}
		{#each currentMessages as message (message.id)}
			<ChatMessage
				{message}
				{lastSentUserId}
				{streamingMsgId}
				{streamingPreAnswer}
			/>
		{/each}

		{#if loading && !streamingMsgId}
			<article class="message assistant pending" aria-busy="true">
				<div class="avatar" aria-hidden="true"><AssistantAvatarGlyph /></div>
				<div class="message-content">
					<div class="message-label">Assistant</div>
					<p class="typing loading-status" role="status" aria-live="polite">
						<span class="spinner" aria-hidden="true"></span>
						<span>{latestStatus || 'Searching sources…'}</span>
					</p>
				</div>
			</article>
		{/if}
	{/if}
</div>

<style>
	.thread {
		flex: 1;
		min-height: 0;
		overflow: auto;
		padding: 0.4rem 0.25rem 1rem;
		scroll-behavior: smooth;
	}

	.message {
		display: grid;
		grid-template-columns: 2.5rem minmax(0, 1fr);
		gap: 0.8rem;
		max-width: 58rem;
		margin: 0 auto 1rem;
	}

	.avatar {
		display: grid;
		place-items: center;
		border-radius: 999px;
		background: var(--chip-bg);
		border: 1px solid var(--chip-border);
		color: var(--chip-fg);
		width: 2.5rem;
		height: 2.5rem;
		font-size: 0.78rem;
	}

	.avatar :global(.avatar-glyph) {
		width: 1.1rem;
		height: 1.1rem;
		display: block;
	}

	.message-content {
		border-radius: 1.2rem;
		padding: 0.9rem 1rem;
		background: var(--msg-assistant-bg);
		border: 1px solid var(--msg-assistant-border);
	}

	.message-label {
		color: var(--text-muted);
		font-size: 0.78rem;
		font-weight: 500;
		margin-bottom: 0.35rem;
	}

	.typing {
		color: var(--text-muted);
	}

	.loading-status {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0;
		min-height: 1.25rem;
	}

	.loading-status .spinner {
		flex-shrink: 0;
	}

	.spinner {
		width: 0.9rem;
		height: 0.9rem;
		border-radius: 999px;
		border: 2px solid var(--spinner-ring);
		border-top-color: var(--spinner-cap);
		animation: spin 0.8s linear infinite;
		flex: 0 0 auto;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	@media (max-width: 640px) {
		.message {
			grid-template-columns: 1fr;
		}

		.avatar {
			display: none;
		}
	}
</style>
