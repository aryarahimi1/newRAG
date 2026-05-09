<script lang="ts">
	import { entityTypes, metadataLabel } from '$lib/recall';
	import { formatMs, formatScore, metadataValue } from '$lib/chat-format';
	import type { PipelineResult, StatusEntry } from '$lib/types/chat';

	let {
		pipelineDetailsOpen = $bindable(false),
		currentStatus,
		currentResult,
		loading,
		compactPipelineUi
	}: {
		pipelineDetailsOpen?: boolean;
		currentStatus: StatusEntry[];
		currentResult: PipelineResult | null;
		loading: boolean;
		compactPipelineUi: boolean;
	} = $props();
</script>

<details class="details-stack" bind:open={pipelineDetailsOpen}>
	<summary class="details-stack-summary">Pipeline &amp; sources</summary>
	<div class="details-stack-body">
		{#if currentStatus.length}
			<details class="detail-panel" open={loading}>
				<summary>Pipeline status</summary>
				<ol class="status-list">
					{#each currentStatus as status (status.id)}
						<li>{status.text}</li>
					{/each}
				</ol>
			</details>
		{/if}

		{#if currentResult}
			<details class="detail-panel">
				<summary>PII redaction</summary>
				<div class="split">
					<div>
						<h3>Original</h3>
						<pre>{currentResult.redaction.original}</pre>
					</div>
					<div>
						<h3>Redacted</h3>
						<pre>{currentResult.redaction.redacted}</pre>
					</div>
				</div>
				{#if currentResult.redaction.entities.length}
					<p class="note warn">
						Detected {currentResult.redaction.entities.length} PII entities: {entityTypes(currentResult)}
					</p>
				{:else}
					<p class="note ok">No PII entities detected.</p>
				{/if}
			</details>

			<details class="detail-panel">
				<summary>Drug detection &amp; auto-ingest</summary>
				{#if currentResult.detected_drugs.length}
					<div class="pill-list">
						{#each currentResult.detected_drugs as drug (`${drug.mention}-${drug.rxcui}`)}
							<span class="data-pill">
								<strong>{drug.canonical}</strong>
								<small>{drug.mention} / RxCUI {drug.rxcui} / {formatScore(drug.score)}</small>
							</span>
						{/each}
					</div>
				{:else}
					<p class="muted">No drug names were detected.</p>
				{/if}

				<div class="ingest-box">
					<p><strong>{currentResult.auto_ingest.skipped ? 'Auto-ingest skipped' : 'Auto-ingest checked'}</strong></p>
					<p class="muted">
						{currentResult.auto_ingest.ingested.length
							? `Ingested ${currentResult.auto_ingest.ingested.join(', ')}`
							: 'No new drug documents were ingested.'}
					</p>
					{#if currentResult.auto_ingest.added_chunks}
						<p class="ok-text">Added {currentResult.auto_ingest.added_chunks} chunks.</p>
					{/if}
					{#if currentResult.auto_ingest.error}
						<p class="warn-text">{currentResult.auto_ingest.error}</p>
					{/if}
				</div>
			</details>

			<details class="detail-panel" open={!compactPipelineUi}>
				<summary>Citations &amp; reranked sources</summary>
				{#if currentResult.reranked.length}
					<div class="source-list">
						{#each currentResult.reranked as source (source.id)}
							<article class="source-card">
								<div class="source-head">
									<strong>{metadataLabel(source)}</strong>
									<span class="score">score {formatScore(source.score)}</span>
								</div>
								<p>{source.text}</p>
								<div class="metadata">
									{#each Object.entries(source.metadata ?? {}) as [key, value] (`${source.id}-${key}`)}
										{#if metadataValue(value)}
											<span>{key}: {metadataValue(value)}</span>
										{/if}
									{/each}
								</div>
							</article>
						{/each}
					</div>
				{:else}
					<p class="muted">No reranked sources returned.</p>
				{/if}
			</details>

			<details class="detail-panel">
				<summary>Generation &amp; timing</summary>
				{#if currentResult.generation}
					<div class="metric-grid">
						<div><strong>{currentResult.generation.model}</strong><span>model</span></div>
						<div><strong>{currentResult.generation.prompt_tokens ?? 'n/a'}</strong><span>prompt tokens</span></div>
						<div>
							<strong>{currentResult.generation.completion_tokens ?? 'n/a'}</strong><span>completion tokens</span>
						</div>
					</div>
				{:else}
					<p class="muted">Generation skipped or unavailable.</p>
				{/if}

				{#if Object.keys(currentResult.timing ?? {}).length}
					<div class="timing-grid">
						{#each Object.entries(currentResult.timing) as [name, value] (name)}
							<div><span>{name}</span><strong>{formatMs(value)}</strong></div>
						{/each}
					</div>
				{/if}
			</details>
		{/if}
	</div>
</details>

<style>
	.details-stack {
		min-width: 0;
	}

	.details-stack-summary {
		list-style: none;
		cursor: pointer;
		font-weight: 600;
		color: var(--summary-color);
		padding: 0.75rem 0.9rem;
		border-radius: 1.1rem;
		border: 1px solid var(--border);
		background: var(--card-bg);
		box-shadow: var(--card-shadow);
	}

	.details-stack-summary::-webkit-details-marker {
		display: none;
	}

	.details-stack-body {
		display: grid;
		gap: 0.65rem;
		margin-top: 0.65rem;
		padding-right: 0.2rem;
		max-height: 34vh;
		overflow: auto;
	}

	@media (max-width: 640px) {
		.details-stack-body {
			max-height: min(50dvh, 22rem);
		}
	}

	@media (min-width: 641px) {
		.details-stack[open] .details-stack-summary {
			display: none;
		}

		.details-stack[open] .details-stack-body {
			margin-top: 0;
		}
	}

	.detail-panel {
		border-radius: 1.1rem;
		padding: 0.85rem 1rem;
		border: 1px solid var(--border);
		background: var(--card-bg);
		box-shadow: var(--card-shadow);
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

	.detail-panel > :global(*:not(summary):first-of-type) {
		margin-top: 0.85rem;
	}

	.split {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.8rem;
	}

	pre {
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		border-radius: 0.8rem;
		padding: 0.8rem;
		color: var(--pre-fg);
		background: var(--pre-bg);
		border: 1px solid var(--pre-border);
		font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
	}

	h3 {
		margin-top: 0;
	}

	.note {
		border-radius: 0.8rem;
		padding: 0.7rem;
		margin-bottom: 0;
	}

	.ok,
	.ok-text {
		color: var(--success);
	}

	.warn,
	.warn-text {
		color: var(--warning);
	}

	.pill-list,
	.metadata {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.data-pill,
	.metadata span {
		border-radius: 999px;
		border: 1px solid var(--border);
		background: var(--data-pill-bg);
		padding: 0.45rem 0.65rem;
	}

	.data-pill {
		display: grid;
		gap: 0.1rem;
	}

	.data-pill small {
		color: var(--text-muted);
	}

	.ingest-box,
	.source-card {
		border-radius: 0.95rem;
		padding: 0.85rem;
		background: var(--ingest-bg);
		border: 1px solid var(--border);
		margin-top: 0.8rem;
	}

	.source-list {
		display: grid;
		gap: 0.7rem;
	}

	.source-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
	}

	.score {
		color: var(--link-cite);
		font-size: 0.82rem;
	}

	.metadata {
		margin-top: 0.65rem;
	}

	.metadata span {
		color: var(--text-muted);
		font-size: 0.78rem;
	}

	.status-list {
		margin-bottom: 0;
		color: var(--text-secondary);
	}

	.status-list li + li {
		margin-top: 0.4rem;
	}

	.timing-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
		gap: 0.55rem;
		margin-top: 0.85rem;
	}

	.timing-grid div {
		display: flex;
		justify-content: space-between;
		gap: 0.75rem;
		padding: 0.55rem 0.65rem;
		border-radius: 0.75rem;
		background: var(--timing-cell-bg);
	}

	.muted {
		color: var(--text-muted);
	}

	@media (max-width: 920px) {
		.split,
		.metric-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
