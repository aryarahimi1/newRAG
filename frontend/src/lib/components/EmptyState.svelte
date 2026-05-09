<script lang="ts">
	import PharmaMark from '$lib/components/icons/PharmaMark.svelte';

	let {
		sampleQuestions,
		loading,
		onAskSample
	}: {
		sampleQuestions: string[];
		loading: boolean;
		onAskSample: (q: string) => void;
	} = $props();
</script>

<section class="empty-state">
	<div class="empty-orb" aria-hidden="true"><PharmaMark /></div>
	<h2>Ask your first question</h2>
	<p>
		Ask about combinations, contraindications, alcohol warnings, or how one medication may affect another.
		Personal details are removed before lookup; answers are drawn from retrieved FDA and NIH sources and shown
		with citations.
	</p>
	<div class="sample-grid" aria-label="Example questions">
		{#each sampleQuestions as sample (sample)}
			<button type="button" class="sample-card" disabled={loading} onclick={() => onAskSample(sample)}>
				{sample}
			</button>
		{/each}
	</div>
</section>

<style>
	.empty-state {
		min-height: 100%;
		display: grid;
		align-content: center;
		justify-items: center;
		text-align: center;
		gap: 1rem;
		max-width: 56rem;
		margin: 0 auto;
		padding: 2rem 1rem;
	}

	.empty-orb {
		display: grid;
		place-items: center;
		border-radius: 999px;
		background: var(--chip-bg);
		border: 1px solid var(--chip-border);
		color: var(--chip-fg);
		font-weight: 800;
		width: 4.5rem;
		height: 4.5rem;
		border-radius: 14px;
		background: var(--empty-orb-bg);
		border: 1px solid var(--empty-orb-border);
		color: var(--chip-fg);
		padding: 0.85rem;
		box-sizing: border-box;
	}

	.empty-orb :global(svg) {
		width: 100%;
		height: 100%;
		display: block;
	}

	h2 {
		margin: 0;
		font-size: clamp(1.8rem, 4vw, 3.4rem);
		letter-spacing: -0.05em;
		color: var(--text-heading);
	}

	p {
		max-width: 43rem;
		color: var(--empty-body);
		margin-top: 0;
	}

	.sample-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 15rem), 1fr));
		gap: 0.75rem;
		width: min(100%, 52rem);
	}

	.sample-card {
		color: var(--sample-card-fg);
		background: var(--sample-card-bg);
		border: 1px solid var(--sample-card-border);
		border-radius: 1rem;
		padding: 0.9rem;
		text-align: left;
		font: inherit;
		cursor: pointer;
		transition:
			transform 0.16s ease,
			border-color 0.16s ease,
			background 0.16s ease;
	}

	.sample-card:hover:not(:disabled) {
		transform: translateY(-1px);
	}

	.sample-card:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	@media (max-width: 920px) {
		.sample-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
