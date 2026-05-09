<script lang="ts">
	import type { RecallAlert } from '$lib/types/chat';

	let { alerts }: { alerts: RecallAlert[] } = $props();
</script>

<div
	class="recall-strip"
	role="group"
	aria-label="FDA drug recalls found in sources for this answer"
>
	<p class="recall-strip-title">FDA recalls in sources</p>
	<div class="recall-chips">
		{#each alerts as alert (alert.id)}
			<a
				class={['recall-chip', `recall-sev-${alert.severity}`].join(' ')}
				href={alert.sourceUrl}
				target="_blank"
				rel="noopener noreferrer"
			>
				<span class="recall-chip-label">{alert.label}</span>
				<span class="recall-chip-drug">{alert.drugName}</span>
			</a>
		{/each}
	</div>
</div>

<style>
	.recall-strip {
		margin-bottom: 0.75rem;
		padding: 0.65rem 0.75rem;
		border-radius: 0.85rem;
		background: var(--recall-strip-bg);
		border: 1px solid var(--recall-strip-border);
	}

	.recall-strip-title {
		margin: 0 0 0.45rem;
		font-size: 0.72rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--text-muted);
	}

	.recall-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 0.45rem;
	}

	.recall-chip {
		display: inline-flex;
		flex-direction: column;
		gap: 0.06rem;
		align-items: flex-start;
		text-decoration: none;
		border-radius: 0.75rem;
		padding: 0.45rem 0.65rem;
		font-size: 0.8rem;
		line-height: 1.25;
		border: 1px solid transparent;
		transition:
			transform 0.15s ease,
			box-shadow 0.15s ease;
	}

	.recall-chip:hover {
		transform: translateY(-1px);
		box-shadow: 0 2px 10px rgba(15, 23, 42, 0.1);
	}

	.recall-chip-label {
		font-weight: 700;
	}

	.recall-chip-drug {
		font-size: 0.74rem;
		font-weight: 500;
		opacity: 0.9;
		text-transform: capitalize;
	}

	.recall-sev-class1 {
		background: var(--recall-class1-bg);
		border-color: var(--recall-class1-border);
		color: var(--recall-class1-fg);
	}

	.recall-sev-class2 {
		background: var(--recall-class2-bg);
		border-color: var(--recall-class2-border);
		color: var(--recall-class2-fg);
	}

	.recall-sev-class3 {
		background: var(--recall-class3-bg);
		border-color: var(--recall-class3-border);
		color: var(--recall-class3-fg);
	}

	.recall-sev-unknown {
		background: var(--recall-unknown-bg);
		border-color: var(--recall-unknown-border);
		color: var(--recall-unknown-fg);
	}
</style>
