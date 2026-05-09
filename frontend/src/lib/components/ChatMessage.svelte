<script lang="ts">
	import AssistantMarkdownBody from '$lib/components/AssistantMarkdownBody.svelte';
	import RecallChips from '$lib/components/RecallChips.svelte';
	import AssistantAvatarGlyph from '$lib/components/icons/AssistantAvatarGlyph.svelte';
	import UserAvatarGlyph from '$lib/components/icons/UserAvatarGlyph.svelte';
	import type { ChatTurn } from '$lib/types/chat';

	let {
		message,
		lastSentUserId,
		streamingMsgId,
		streamingPreAnswer
	}: {
		message: ChatTurn;
		lastSentUserId: string | null;
		streamingMsgId: string | null;
		streamingPreAnswer: boolean;
	} = $props();

	let articleClass = $derived(
		[
			'message',
			message.role,
			message.id === lastSentUserId && 'sent-flash',
			message.id === streamingMsgId && 'streaming',
			message.id === streamingMsgId && streamingPreAnswer && 'pre-answer'
		]
			.filter(Boolean)
			.join(' ')
	);
</script>

<article class={articleClass}>
	<div class="avatar" aria-hidden="true">
		{#if message.role === 'user'}
			<UserAvatarGlyph />
		{:else}
			<AssistantAvatarGlyph />
		{/if}
	</div>
	<div class="message-content">
		<div class="message-label">{message.role === 'user' ? 'You' : 'Assistant'}</div>
		{#if message.role === 'assistant'}
			{#if message.recallAlerts?.length}
				<RecallChips alerts={message.recallAlerts} />
			{/if}
			<AssistantMarkdownBody content={message.content} />
		{:else}
			<p>{message.content}</p>
		{/if}
	</div>
</article>

<style>
	.message {
		display: grid;
		grid-template-columns: 2.5rem minmax(0, 1fr);
		gap: 0.8rem;
		max-width: 58rem;
		margin: 0 auto 1rem;
	}

	.message.user {
		max-width: 50rem;
		margin-left: auto;
		margin-right: max(0rem, calc((100% - 58rem) / 2));
	}

	.avatar {
		display: grid;
		place-items: center;
		border-radius: 999px;
		background: var(--chip-bg);
		border: 1px solid var(--chip-border);
		color: var(--chip-fg);
		font-weight: 800;
		width: 2.5rem;
		height: 2.5rem;
		font-size: 0.78rem;
	}

	.avatar :global(.avatar-glyph) {
		width: 1.1rem;
		height: 1.1rem;
		display: block;
	}

	.message.user .avatar {
		background: var(--msg-user-avatar-bg);
		color: var(--msg-user-avatar-fg);
	}

	.message-content {
		border-radius: 1.2rem;
		padding: 0.9rem 1rem;
		background: var(--msg-assistant-bg);
		border: 1px solid var(--msg-assistant-border);
	}

	.message.user .message-content {
		background: var(--msg-user-bg);
		border-color: var(--msg-user-border);
	}

	.message.pre-answer .message-content {
		border-color: var(--pre-stream-border);
	}

	.message-label {
		color: var(--text-muted);
		font-size: 0.78rem;
		font-weight: 500;
		margin-bottom: 0.35rem;
	}

	.message-content p {
		margin: 0;
		white-space: pre-wrap;
	}

	.sent-flash .message-content {
		animation: flash 0.7s ease;
	}

	@keyframes flash {
		0% {
			box-shadow: 0 0 0 0 rgba(30, 64, 175, 0.38);
		}
		100% {
			box-shadow: 0 0 0 14px rgba(30, 64, 175, 0);
		}
	}

	@media (max-width: 640px) {
		.message,
		.message.user {
			grid-template-columns: 1fr;
			margin-right: 0;
		}

		.avatar {
			display: none;
		}
	}
</style>
