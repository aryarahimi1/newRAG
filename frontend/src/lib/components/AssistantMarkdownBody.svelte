<script lang="ts">
	import { sanitizeMarkdownHtml } from '$lib/markdown';

	let { content }: { content: string } = $props();

	let html = $derived(sanitizeMarkdownHtml(content));
</script>

<div class="answer-body">{@html html}</div>

<style>
	.answer-body :global(p) {
		margin: 0 0 0.85rem;
	}

	.answer-body :global(p:last-child) {
		margin-bottom: 0;
	}

	.answer-body :global(.cite) {
		color: var(--link-cite);
		font-weight: 600;
	}

	.answer-body :global(.md-section h4) {
		margin: 0 0 0.35rem;
		color: var(--md-heading);
	}

	.answer-body :global(pre) {
		overflow-x: auto;
		margin: 0.5rem 0;
		padding: 0.65rem 0.75rem;
		border-radius: 0.65rem;
		background: var(--pre-bg);
		border: 1px solid var(--pre-border);
		font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
		font-size: 0.85em;
	}

	.answer-body :global(code) {
		font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
		font-size: 0.9em;
	}

	.answer-body :global(a) {
		color: var(--link-cite);
		text-decoration: underline;
		text-underline-offset: 2px;
	}
</style>
