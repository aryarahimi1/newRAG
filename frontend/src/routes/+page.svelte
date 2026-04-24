<script lang="ts">
	type ChatTurn = { role: 'user' | 'assistant'; content: string };

	type CorpusStats = {
		n_chunks: number;
		n_drugs: number;
		n_sources: number;
		embedding_model: string;
		collection: string;
		drugs?: string[];
	};

	type PipelineResult = {
		question: string;
		redaction: {
			original: string;
			redacted: string;
			entities: { entity_type: string }[];
		};
		detected_drugs: { mention: string; canonical: string; rxcui: string; score: number }[];
		auto_ingest: {
			missing: { mention: string; canonical: string; rxcui: string; score: number }[];
			ingested: string[];
			added_chunks: number;
			error: string | null;
			skipped: boolean;
		};
		retrieved: { id: string; text: string; metadata: Record<string, string>; score: number }[];
		reranked: { id: string; text: string; metadata: Record<string, string>; score: number }[];
		generation: {
			answer: string;
			model: string;
			prompt_tokens: number | null;
			completion_tokens: number | null;
		} | null;
		timing: Record<string, number>;
		error: string | null;
		status_log?: string[];
	};

	const sampleQuestions = [
		'Can I take ibuprofen with lisinopril for my blood pressure?',
		'Is it safe to combine warfarin and aspirin?',
		"I'm John Smith (john@example.com). Does metformin interact with alcohol?",
		'What happens if I take sertraline and tramadol together?',
		'Can I take omeprazole while on clopidogrel?',
		'Does rifampin reduce the effectiveness of warfarin?'
	];

	let messages = $state<ChatTurn[]>([]);
	let lastResult = $state<PipelineResult | null>(null);
	let corpus = $state<CorpusStats | null>(null);
	let apiConfig = $state<{
		openrouter_model: string;
		pii_backend: string;
		has_openrouter_key: boolean;
	} | null>(null);

	let topKRetrieve = $state(20);
	let topKRerank = $state(5);
	let autoIngest = $state(true);
	let skipGeneration = $state(false);

	let input = $state('');
	let loading = $state(false);
	let errorMsg = $state<string | null>(null);
	let statusLines = $state<string[]>([]);

	$effect(() => {
		if (apiConfig && !apiConfig.has_openrouter_key) {
			skipGeneration = true;
		}
	});

	function segmentAnswer(text: string): { kind: 'text' | 'cite'; value: string }[] {
		const re = /\[(\d+)\]/g;
		const out: { kind: 'text' | 'cite'; value: string }[] = [];
		let last = 0;
		let m: RegExpExecArray | null;
		while ((m = re.exec(text)) !== null) {
			if (m.index > last) {
				out.push({ kind: 'text', value: text.slice(last, m.index) });
			}
			out.push({ kind: 'cite', value: m[1] ?? '' });
			last = m.index + m[0].length;
		}
		if (last < text.length) {
			out.push({ kind: 'text', value: text.slice(last) });
		}
		return out.length ? out : [{ kind: 'text', value: text }];
	}

	async function loadMeta() {
		try {
			const [s, c] = await Promise.all([
				fetch('/api/corpus/stats').then((r) => r.json()),
				fetch('/api/config').then((r) => r.json())
			]);
			corpus = s;
			apiConfig = c;
			if (!c.has_openrouter_key) skipGeneration = true;
		} catch {
			corpus = null;
		}
	}

	$effect(() => {
		void loadMeta();
	});

	function clearChat() {
		messages = [];
		lastResult = null;
		errorMsg = null;
		statusLines = [];
	}

	function buildHistory(): ChatTurn[] {
		const h: ChatTurn[] = [];
		for (let i = 0; i + 1 < messages.length; i += 2) {
			const u = messages[i];
			const a = messages[i + 1];
			if (u?.role === 'user' && a?.role === 'assistant') {
				h.push(u, a);
			}
		}
		return h;
	}

	async function send(question: string) {
		const q = question.trim();
		if (!q || loading) return;

		errorMsg = null;
		statusLines = [];
		messages = [...messages, { role: 'user', content: q }];
		input = '';
		loading = true;

		try {
			const history = buildHistory();
			const res = await fetch('/api/chat', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					question: q,
					history,
					top_k_retrieve: topKRetrieve,
					top_k_rerank: topKRerank,
					auto_ingest: autoIngest,
					skip_generation: skipGeneration
				})
			});
			const data = (await res.json()) as PipelineResult & { detail?: string };
			if (!res.ok) {
				throw new Error(typeof data.detail === 'string' ? data.detail : res.statusText);
			}
			lastResult = data;
			if (data.status_log?.length) statusLines = data.status_log;

			let answerText =
				data.generation?.answer ??
				(data.error ? `_(generation failed: ${data.error})_` : '_(LLM call skipped — retrieval only)_');

			messages = [...messages, { role: 'assistant', content: answerText }];
			void loadMeta();
		} catch (e) {
			errorMsg = e instanceof Error ? e.message : 'Request failed';
			messages = messages.slice(0, -1);
		} finally {
			loading = false;
		}
	}

	function onSubmit(e: Event) {
		e.preventDefault();
		void send(input);
	}

	function useSample(s: string) {
		void send(s);
	}
</script>

<svelte:head>
	<title>Drug Interaction RAG</title>
	<meta
		name="description"
		content="Grounded drug interaction Q&A with PII redaction, RxNorm detection, on-the-fly ingest, and citations."
	/>
</svelte:head>

<div class="shell">
	<aside class="sidebar glass">
		<div class="brand">
			<span class="logo" aria-hidden="true">◇</span>
			<div>
				<h1 class="title">Drug RAG</h1>
				<p class="tagline">DailyMed · MedlinePlus · Chroma</p>
			</div>
		</div>

		{#if corpus}
			<section class="stats">
				<h2>Corpus</h2>
				<div class="stat-grid">
					<div class="stat">
						<span class="stat-val">{corpus.n_chunks.toLocaleString()}</span>
						<span class="stat-lbl">chunks</span>
					</div>
					<div class="stat">
						<span class="stat-val">{corpus.n_drugs}</span>
						<span class="stat-lbl">drugs</span>
					</div>
					<div class="stat">
						<span class="stat-val">{corpus.n_sources}</span>
						<span class="stat-lbl">sources</span>
					</div>
				</div>
				<p class="fineprint mono">{corpus.embedding_model}</p>
			</section>
		{:else}
			<p class="fineprint">Corpus stats unavailable — is the API running?</p>
		{/if}

		<section class="controls">
			<h2>Pipeline</h2>
			<label class="field">
				<span>Retrieve top-k</span>
				<input type="range" min="5" max="50" step="5" bind:value={topKRetrieve} />
				<span class="mono">{topKRetrieve}</span>
			</label>
			<label class="field">
				<span>Rerank top-k</span>
				<input type="range" min="1" max="10" step="1" bind:value={topKRerank} />
				<span class="mono">{topKRerank}</span>
			</label>
			<label class="toggle">
				<input type="checkbox" bind:checked={autoIngest} />
				<span
					>Auto-ingest unknown drugs
					<span class="hint" title="RxNorm detects names missing from Chroma; FDA + MedlinePlus fetch runs before retrieval."
						>ⓘ</span
					></span
				>
			</label>
			<label class="toggle">
				<input type="checkbox" bind:checked={skipGeneration} />
				<span>Skip LLM (retrieval only)</span>
			</label>
		</section>

		{#if apiConfig}
			<section class="meta">
				<p class="fineprint">PII: <span class="mono">{apiConfig.pii_backend}</span></p>
				<p class="fineprint mono">{apiConfig.openrouter_model}</p>
			</section>
		{/if}

		<section class="samples">
			<h2>Try asking</h2>
			<div class="chips">
				{#each sampleQuestions as s}
					<button type="button" class="chip" onclick={() => useSample(s)}>{s}</button>
				{/each}
			</div>
		</section>

		<button type="button" class="ghost-btn" onclick={clearChat}>Clear conversation</button>

		<p class="disclaimer">
			Educational demo only — not medical advice. Grounded in FDA / NIH public sources; always confirm with a
			clinician.
		</p>
	</aside>

	<main class="main">
		<header class="main-head glass">
			<div>
				<h2 class="main-title">Conversation</h2>
				<p class="main-sub">
					Multi-turn Q&A — PII redacted before search. Unknown drugs trigger on-the-fly ingest (RxNorm → Chroma),
					then hybrid retrieval and reranking.
				</p>
			</div>
			{#if loading}
				<div class="pulse" role="status">Running pipeline…</div>
			{/if}
		</header>

		{#if errorMsg}
			<div class="banner error" role="alert">{errorMsg}</div>
		{/if}

		<div class="thread">
			{#each messages as m, i (i)}
				<article class="bubble {m.role}">
					<span class="role">{m.role === 'user' ? 'You' : 'Assistant'}</span>
					<div class="bubble-body">
						{#if m.role === 'assistant'}
							<p class="answer">
								{#each segmentAnswer(m.content) as part, pi (part.kind + part.value + pi)}
									{#if part.kind === 'cite'}
										<span class="cite">[{part.value}]</span>
									{:else}
										{part.value}
									{/if}
								{/each}
							</p>
						{:else}
							<p class="answer">{m.content}</p>
						{/if}
					</div>
				</article>
			{/each}
		</div>

		{#if lastResult}
			<div class="panels glass">
				{#if statusLines.length}
					<section class="panel status-panel">
						<h3>Pipeline trace</h3>
						<ol class="status-list">
							{#each statusLines as line, j}
								<li>{line}</li>
							{/each}
						</ol>
					</section>
				{/if}

				<section class="panel">
					<h3>1 · PII redaction</h3>
					<div class="split">
						<div>
							<h4>Original</h4>
							<pre class="code">{lastResult.redaction.original}</pre>
						</div>
						<div>
							<h4>Redacted</h4>
							<pre class="code">{lastResult.redaction.redacted}</pre>
						</div>
					</div>
					{#if lastResult.redaction.entities?.length}
						<p class="note warn">
							Detected {lastResult.redaction.entities.length} PII entit{lastResult.redaction.entities.length === 1
								? 'y'
								: 'ies'}: {[...new Set(lastResult.redaction.entities.map((e) => e.entity_type))].join(', ')}
						</p>
					{:else}
						<p class="note ok">No PII detected.</p>
					{/if}
				</section>

				<section class="panel">
					<h3>2 · Drug detection &amp; auto-ingest</h3>
					{#if !lastResult.detected_drugs?.length}
						<p class="note">No RxNorm drug hits — retrieval runs over the full corpus.</p>
					{:else}
						<div class="drug-row">
							{#each lastResult.detected_drugs as d}
								<div class="drug-card">
									<strong>{d.canonical}</strong>
									<span class="fineprint mono">RxCUI {d.rxcui} · {d.score.toFixed(0)}/100</span>
								</div>
							{/each}
						</div>
					{/if}
					{#if lastResult.auto_ingest.skipped}
						<p class="fineprint">Auto-ingest disabled.</p>
					{:else if lastResult.auto_ingest.missing?.length}
						{#if lastResult.auto_ingest.added_chunks > 0}
							<p class="note ok">
								Learned {#each lastResult.auto_ingest.missing as m, idx}{idx > 0 ? ', ' : ''}<strong
									>{m.canonical}</strong
								>{/each} on the fly — added {lastResult.auto_ingest.added_chunks} chunks to Chroma.
							</p>
						{:else if lastResult.auto_ingest.error}
							<p class="note warn">Ingest note: {lastResult.auto_ingest.error}</p>
						{:else}
							<p class="note">Tried to ingest missing drugs but no upstream documents returned.</p>
						{/if}
					{:else}
						<p class="fineprint">All detected drugs already indexed.</p>
					{/if}
				</section>

				<section class="panel">
					<h3>3 · Answer</h3>
					{#if lastResult.error}
						<p class="note err">Generation failed: {lastResult.error}</p>
					{:else if lastResult.generation}
						<p class="fineprint mono">
							{lastResult.generation.model}
							{#if lastResult.generation.prompt_tokens != null}
								· {lastResult.generation.prompt_tokens} prompt / {lastResult.generation.completion_tokens}
								completion tokens
							{/if}
						</p>
					{:else}
						<p class="note">LLM skipped.</p>
					{/if}
				</section>

				<section class="panel">
					<h3>4 · Citations (reranked)</h3>
					{#if !lastResult.reranked?.length}
						<p class="note">No passages retrieved.</p>
					{:else}
						<div class="citations">
							{#each lastResult.reranked as c, idx (c.id)}
								<details class="cite-card" open={idx === 0}>
									<summary>
										<span class="badge">{idx + 1}</span>
										<span class="cite-title"
											>{c.metadata?.drug_name ?? 'unknown'} — {c.metadata?.section ?? ''} · {c.score.toFixed(
												3
											)}</span
										>
									</summary>
									{#if c.metadata?.source_url}
										<p class="fineprint">
											<a href={c.metadata.source_url} target="_blank" rel="noreferrer"
												>{c.metadata.source ?? c.metadata.source_url}</a
											>
										</p>
									{/if}
									<pre class="chunk">{c.text}</pre>
								</details>
							{/each}
						</div>
					{/if}
				</section>

				<section class="panel">
					<details>
						<summary>Debug: pre-rerank + timings</summary>
						<pre class="code mono">{JSON.stringify(
								{
									redact_ms: Math.round(lastResult.timing.redact_ms * 10) / 10,
									detect_ms: Math.round(lastResult.timing.detect_ms * 10) / 10,
									ingest_ms: Math.round(lastResult.timing.ingest_ms * 10) / 10,
									retrieve_ms: Math.round(lastResult.timing.retrieve_ms * 10) / 10,
									rerank_ms: Math.round(lastResult.timing.rerank_ms * 10) / 10,
									generate_ms: Math.round(lastResult.timing.generate_ms * 10) / 10,
									total_ms: Math.round(lastResult.timing.total_ms * 10) / 10
								},
								null,
								2
							)}</pre>
						<div class="citations">
							{#each lastResult.retrieved as c, idx}
								<div class="cite-card flat">
									<span class="badge dim">R{idx + 1}</span>
									<span class="cite-title">{c.score.toFixed(3)} · {c.metadata?.drug_name}</span>
									<pre class="chunk">{c.text}</pre>
								</div>
							{/each}
						</div>
					</details>
				</section>
			</div>
		{/if}

		<form class="composer glass" onsubmit={onSubmit}>
			<input
				class="composer-input"
				type="text"
				placeholder="Ask about drug interactions…"
				bind:value={input}
				disabled={loading}
				autocomplete="off"
			/>
			<button type="submit" class="send" disabled={loading || !input.trim()}>Send</button>
		</form>
	</main>
</div>

<style>
	.shell {
		display: grid;
		grid-template-columns: minmax(280px, 340px) 1fr;
		min-height: 100vh;
		gap: 0;
	}
	@media (max-width: 960px) {
		.shell {
			grid-template-columns: 1fr;
		}
		.sidebar {
			border-right: none;
			border-bottom: 1px solid var(--border);
		}
	}

	.glass {
		background: var(--bg-panel);
		backdrop-filter: blur(16px);
		border: 1px solid var(--border);
	}

	.sidebar {
		padding: 1.5rem;
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
		border-right: 1px solid var(--border);
	}

	.brand {
		display: flex;
		gap: 0.75rem;
		align-items: center;
	}
	.logo {
		font-size: 1.75rem;
		color: var(--accent);
		text-shadow: 0 0 24px var(--accent-dim);
	}
	.title {
		margin: 0;
		font-size: 1.35rem;
		font-weight: 700;
		letter-spacing: -0.02em;
	}
	.tagline {
		margin: 0.15rem 0 0;
		font-size: 0.78rem;
		color: var(--text-muted);
	}

	h2,
	h3,
	h4 {
		margin: 0 0 0.5rem;
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.12em;
		color: var(--text-muted);
		font-weight: 600;
	}
	h3 {
		font-size: 0.8rem;
		margin-bottom: 0.75rem;
		color: var(--accent);
	}

	.stats .stat-grid {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 0.5rem;
	}
	.stat {
		background: var(--bg-elevated);
		border-radius: 10px;
		padding: 0.6rem 0.5rem;
		text-align: center;
		border: 1px solid var(--border);
	}
	.stat-val {
		display: block;
		font-weight: 700;
		font-size: 1.1rem;
		color: var(--accent-hot);
	}
	.stat-lbl {
		font-size: 0.65rem;
		color: var(--text-muted);
		text-transform: uppercase;
	}

	.field {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 0.35rem 0.75rem;
		align-items: center;
		font-size: 0.82rem;
		margin-bottom: 0.65rem;
	}
	.field span:first-child {
		grid-column: 1 / -1;
		color: var(--text-muted);
	}
	.field input[type='range'] {
		grid-column: 1 / 2;
		width: 100%;
		accent-color: var(--accent);
	}

	.toggle {
		display: flex;
		align-items: flex-start;
		gap: 0.5rem;
		font-size: 0.84rem;
		margin-bottom: 0.5rem;
		cursor: pointer;
	}
	.toggle input {
		margin-top: 0.2rem;
		accent-color: var(--accent);
	}
	.hint {
		cursor: help;
		opacity: 0.6;
	}

	.chips {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		max-height: 220px;
		overflow-y: auto;
	}
	.chip {
		text-align: left;
		font: inherit;
		font-size: 0.78rem;
		padding: 0.45rem 0.55rem;
		border-radius: 8px;
		border: 1px solid var(--border);
		background: var(--bg-elevated);
		color: var(--text);
		cursor: pointer;
		transition:
			border-color 0.15s,
			background 0.15s;
	}
	.chip:hover {
		border-color: var(--accent);
		background: var(--accent-dim);
	}

	.ghost-btn {
		margin-top: auto;
		font: inherit;
		font-size: 0.82rem;
		padding: 0.55rem;
		border-radius: 8px;
		border: 1px dashed var(--border);
		background: transparent;
		color: var(--text-muted);
		cursor: pointer;
	}
	.ghost-btn:hover {
		color: var(--text);
		border-color: var(--text-muted);
	}

	.disclaimer {
		font-size: 0.72rem;
		color: var(--text-muted);
		line-height: 1.4;
		margin: 0;
	}

	.main {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
		padding: 1rem 1.25rem 1.25rem;
		gap: 1rem;
	}

	.main-head {
		padding: 1rem 1.25rem;
		border-radius: 14px;
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 1rem;
		flex-wrap: wrap;
	}
	.main-title {
		margin: 0;
		font-size: 1.25rem;
		font-weight: 600;
		letter-spacing: -0.02em;
	}
	.main-sub {
		margin: 0.35rem 0 0;
		font-size: 0.85rem;
		color: var(--text-muted);
		max-width: 62ch;
	}

	.pulse {
		font-size: 0.82rem;
		color: var(--accent);
		animation: glow 1.2s ease-in-out infinite alternate;
	}
	@keyframes glow {
		from {
			opacity: 0.5;
		}
		to {
			opacity: 1;
		}
	}

	.banner {
		padding: 0.65rem 1rem;
		border-radius: 10px;
		font-size: 0.88rem;
	}
	.banner.error {
		background: rgba(251, 113, 133, 0.12);
		border: 1px solid rgba(251, 113, 133, 0.35);
		color: #fecdd3;
	}

	.thread {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 0.85rem;
		overflow-y: auto;
		padding-bottom: 0.5rem;
	}

	.bubble {
		max-width: min(720px, 100%);
		align-self: flex-start;
		padding: 0.85rem 1rem;
		border-radius: 14px 14px 14px 4px;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
	}
	.bubble.user {
		align-self: flex-end;
		border-radius: 14px 14px 4px 14px;
		background: linear-gradient(135deg, rgba(94, 234, 212, 0.12), rgba(244, 114, 182, 0.08));
	}
	.role {
		font-size: 0.65rem;
		text-transform: uppercase;
		letter-spacing: 0.14em;
		color: var(--text-muted);
	}
	.answer {
		margin: 0.35rem 0 0;
		white-space: pre-wrap;
		font-size: 0.95rem;
	}
	.cite {
		display: inline-block;
		margin: 0 0.1rem;
		padding: 0.05rem 0.35rem;
		border-radius: 4px;
		background: var(--accent-dim);
		color: var(--accent);
		font-weight: 600;
		font-size: 0.85em;
		vertical-align: baseline;
	}

	.panels {
		border-radius: 14px;
		padding: 1rem 1.15rem;
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
		max-height: 48vh;
		overflow-y: auto;
	}
	.panel h4 {
		font-size: 0.68rem;
		margin-top: 0.5rem;
	}
	.split {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.75rem;
	}
	@media (max-width: 700px) {
		.split {
			grid-template-columns: 1fr;
		}
	}
	.code {
		margin: 0;
		padding: 0.65rem;
		border-radius: 8px;
		background: rgba(0, 0, 0, 0.35);
		border: 1px solid var(--border);
		overflow-x: auto;
		white-space: pre-wrap;
		word-break: break-word;
	}
	.note {
		margin: 0.5rem 0 0;
		font-size: 0.86rem;
	}
	.note.ok {
		color: var(--success);
	}
	.note.warn {
		color: var(--warning);
	}
	.note.err {
		color: var(--danger);
	}

	.drug-row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}
	.drug-card {
		padding: 0.45rem 0.65rem;
		border-radius: 8px;
		border: 1px solid var(--border);
		background: rgba(0, 0, 0, 0.25);
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}

	.citations {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.cite-card {
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 0.5rem 0.65rem;
		background: rgba(0, 0, 0, 0.2);
	}
	.cite-card.flat {
		padding-top: 0.65rem;
	}
	.cite-card summary {
		cursor: pointer;
		list-style: none;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-weight: 500;
	}
	.cite-card summary::-webkit-details-marker {
		display: none;
	}
	.badge {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: 1.5rem;
		height: 1.5rem;
		border-radius: 6px;
		background: var(--accent);
		color: #042f2e;
		font-size: 0.75rem;
		font-weight: 700;
	}
	.badge.dim {
		background: var(--text-muted);
		color: var(--bg-deep);
	}
	.cite-title {
		font-size: 0.82rem;
	}
	.chunk {
		margin: 0.5rem 0 0;
		font-size: 0.78rem;
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 200px;
		overflow-y: auto;
	}

	.status-panel h3 {
		color: var(--accent-hot);
	}
	.status-list {
		margin: 0;
		padding-left: 1.1rem;
		font-size: 0.82rem;
		color: var(--text-muted);
	}

	.composer {
		display: flex;
		gap: 0.65rem;
		padding: 0.55rem 0.65rem;
		border-radius: 12px;
		margin-top: auto;
	}
	.composer-input {
		flex: 1;
		border: none;
		background: rgba(0, 0, 0, 0.35);
		border-radius: 8px;
		padding: 0.65rem 0.85rem;
		color: var(--text);
		font: inherit;
		font-size: 0.95rem;
		outline: none;
		border: 1px solid transparent;
	}
	.composer-input:focus {
		border-color: var(--accent);
	}
	.send {
		font: inherit;
		font-weight: 600;
		padding: 0 1.25rem;
		border: none;
		border-radius: 8px;
		cursor: pointer;
		background: linear-gradient(120deg, var(--accent), #2dd4bf);
		color: #042f2e;
	}
	.send:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}

	.fineprint {
		font-size: 0.72rem;
		color: var(--text-muted);
		margin: 0.25rem 0 0;
	}
	.mono {
		font-family: 'JetBrains Mono', ui-monospace, monospace;
		font-size: 0.72em;
	}
</style>
