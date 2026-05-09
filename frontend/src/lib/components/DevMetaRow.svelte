<script lang="ts">
	import type { CorpusStats } from '$lib/types/chat';

	let {
		corpus,
		apiConfig
	}: {
		corpus: CorpusStats | null;
		apiConfig: {
			openrouter_model: string;
			pii_backend: string;
			has_openrouter_key: boolean;
		} | null;
	} = $props();
</script>

<section class="meta-row" aria-label="Corpus and model metadata">
	<details class="meta-card">
		<summary>Corpus</summary>
		{#if corpus}
			<div class="metric-grid">
				<div><strong>{corpus.n_chunks.toLocaleString()}</strong><span>chunks</span></div>
				<div><strong>{corpus.n_drugs.toLocaleString()}</strong><span>drugs</span></div>
				<div><strong>{corpus.n_sources.toLocaleString()}</strong><span>sources</span></div>
			</div>
			<p class="mono muted">{corpus.collection} / {corpus.embedding_model}</p>
		{:else}
			<p class="muted">Corpus stats unavailable. Check that the API is running.</p>
		{/if}
	</details>

	<details class="meta-card">
		<summary>Model &amp; privacy</summary>
		{#if apiConfig}
			<p><span class="muted">Generation</span> <strong>{apiConfig.openrouter_model}</strong></p>
			<p><span class="muted">PII backend</span> <strong>{apiConfig.pii_backend}</strong></p>
			<p class={apiConfig.has_openrouter_key ? 'ok-text' : 'warn-text'}>
				{apiConfig.has_openrouter_key ? 'OpenRouter key available' : 'No OpenRouter key; retrieval-only mode enabled'}
			</p>
		{:else}
			<p class="muted">Config unavailable.</p>
		{/if}
	</details>
</section>

<style>
	.meta-row {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.8rem;
	}

	.meta-card {
		border: 1px solid var(--border);
		background: var(--card-bg);
		box-shadow: var(--card-shadow);
		border-radius: 1.1rem;
		padding: 0.9rem;
	}

	summary {
		cursor: pointer;
		font-weight: 500;
		color: var(--summary-color);
		font: inherit;
	}

	.metric-grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 0.65rem;
		margin-top: 0.85rem;
	}

	.metric-grid div {
		padding: 0.65rem;
		border-radius: 0.8rem;
		background: var(--metric-cell-bg);
		border: 1px solid var(--border);
		display: grid;
		gap: 0.2rem;
	}

	.metric-grid strong {
		overflow-wrap: anywhere;
	}

	.metric-grid span {
		color: var(--text-muted);
		font-size: 0.78rem;
	}

	.mono {
		font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
	}

	.muted {
		color: var(--text-muted);
	}

	.ok-text {
		color: var(--success);
	}

	.warn-text {
		color: var(--warning);
	}

	p {
		margin-top: 0;
	}

	@media (max-width: 920px) {
		.meta-row {
			grid-template-columns: 1fr;
		}
	}
</style>
