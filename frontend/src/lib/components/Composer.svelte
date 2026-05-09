<script lang="ts">
	let {
		input = $bindable(''),
		loading,
		devModeEnabled,
		controlsOpen = $bindable(false),
		topKRetrieve = $bindable(20),
		topKRerank = $bindable(5),
		autoIngest = $bindable(true),
		skipGeneration = $bindable(false),
		apiHasKey,
		onSubmit,
		onComposerKeydown,
		onControlsToggle
	}: {
		input?: string;
		loading: boolean;
		devModeEnabled: boolean;
		controlsOpen?: boolean;
		topKRetrieve?: number;
		topKRerank?: number;
		autoIngest?: boolean;
		skipGeneration?: boolean;
		apiHasKey: boolean | undefined;
		onSubmit: (event: SubmitEvent) => void;
		onComposerKeydown: (event: KeyboardEvent) => void;
		onControlsToggle: (event: ToggleEvent & { currentTarget: HTMLDetailsElement }) => void;
	} = $props();
</script>

<div class={['composer', controlsOpen && devModeEnabled && 'composer-drawer-active'].filter(Boolean).join(' ')}>
	<form class="composer-form" onsubmit={onSubmit}>
		<div class="composer-main">
			<label class="sr-only" for="question-input">Ask a drug interaction question</label>
			<textarea
				id="question-input"
				bind:value={input}
				onkeydown={onComposerKeydown}
				placeholder="Ask about a drug combination..."
				rows="2"
				disabled={loading}
			></textarea>
			<button type="submit" class="primary send" disabled={loading || !input.trim()}>
				{loading ? 'Sending…' : 'Send'}
			</button>
		</div>
	</form>

	{#if devModeEnabled}
		<div class="composer-footer">
			<details class="composer-drawer" bind:open={controlsOpen} ontoggle={onControlsToggle}>
				<summary>Controls</summary>
				<div class="control-grid">
					<label>
						<span>Retrieve top-k</span>
						<input type="range" min="5" max="50" step="5" bind:value={topKRetrieve} />
						<output>{topKRetrieve}</output>
					</label>
					<label>
						<span>Rerank top-k</span>
						<input type="range" min="1" max="10" step="1" bind:value={topKRerank} />
						<output>{topKRerank}</output>
					</label>
					<label class="check-row">
						<input type="checkbox" bind:checked={autoIngest} />
						<span>Auto-ingest unknown drugs</span>
					</label>
					<label class="check-row">
						<input type="checkbox" bind:checked={skipGeneration} disabled={apiHasKey === false} />
						<span>Skip LLM</span>
					</label>
				</div>
			</details>
		</div>
	{/if}
</div>

<style>
	.composer {
		border: 1px solid var(--border);
		background: var(--card-bg);
		box-shadow: var(--card-shadow);
		border-radius: 1.35rem;
		padding: 0.75rem;
	}

	.composer-form {
		margin: 0;
		border: 0;
		padding: 0;
		background: transparent;
	}

	.composer-main {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	textarea {
		flex: 1;
		min-width: 0;
		resize: none;
		outline: none;
		border-radius: 1rem;
		padding: 0.9rem 1rem;
		background: var(--input-bg);
		color: var(--input-fg);
		font: inherit;
		line-height: 1.45;
		border: 1px solid var(--border);
	}

	textarea::placeholder {
		color: var(--placeholder);
	}

	.primary {
		background: var(--btn-primary);
		color: var(--brand-mark-fg);
		font-weight: 600;
		padding: 0.8rem 1.05rem;
		border-radius: 999px;
		border: 0;
		cursor: pointer;
		font: inherit;
		transition:
			transform 0.16s ease,
			background 0.16s ease;
	}

	.primary:hover:not(:disabled) {
		background: var(--btn-primary-hover);
		transform: translateY(-1px);
	}

	.primary:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	.send {
		align-self: stretch;
		min-width: 6.5rem;
	}

	.composer-footer {
		display: flex;
		align-items: flex-start;
		gap: 0.6rem;
		flex-wrap: wrap;
		margin-top: 0.6rem;
	}

	.composer-drawer {
		position: relative;
	}

	.composer-drawer summary {
		list-style: none;
		border-radius: 999px;
		padding: 0.5rem 0.75rem;
		background: var(--composer-drawer-summary-bg);
		border: 1px solid var(--composer-drawer-summary-border);
		cursor: pointer;
		font: inherit;
		font-weight: 500;
		color: var(--summary-color);
	}

	.composer-drawer summary::-webkit-details-marker {
		display: none;
	}

	.composer-drawer[open] summary {
		border-color: var(--composer-drawer-open-border);
	}

	.control-grid {
		position: absolute;
		bottom: calc(100% + 0.6rem);
		left: 0;
		z-index: 10;
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.85rem;
		width: min(28rem, 82vw);
		max-height: 50vh;
		overflow-y: auto;
		border-radius: 1rem;
		border: 1px solid var(--border);
		background: var(--card-bg);
		box-shadow: var(--shadow-popover);
		padding: 0.85rem;
	}

	.control-grid label {
		display: grid;
		gap: 0.35rem;
		color: var(--tab-fg);
	}

	.control-grid output {
		color: var(--accent);
		font-weight: 600;
	}

	input[type='range'] {
		accent-color: var(--btn-primary);
	}

	input[type='checkbox'] {
		width: 1rem;
		height: 1rem;
		accent-color: var(--btn-primary);
	}

	.check-row {
		grid-template-columns: auto 1fr;
		align-items: center;
	}

	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}

	@media (max-width: 920px) {
		.composer.composer-drawer-active {
			position: relative;
			z-index: 45;
		}

		.composer-drawer[open] .control-grid {
			z-index: 46;
		}

		.control-grid {
			grid-template-columns: 1fr;
		}
	}

	@media (max-width: 640px) {
		.composer-main {
			align-items: stretch;
			flex-direction: column;
		}

		.send {
			min-height: 2.8rem;
		}
	}
</style>
