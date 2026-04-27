<script lang="ts">
	import { tick } from 'svelte';

	type Role = 'user' | 'assistant';
	type ChatTurn = { id: string; role: Role; content: string };
	type StatusEntry = { id: string; text: string };

	type CorpusStats = {
		n_chunks: number;
		n_drugs: number;
		n_sources: number;
		embedding_model: string;
		collection: string;
		drugs?: string[];
	};

	type MetadataValue = string | number | boolean | null | undefined;
	type Source = {
		id: string;
		text: string;
		metadata: Record<string, MetadataValue>;
		score: number;
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
		retrieved: Source[];
		reranked: Source[];
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

	type ChatSession = {
		id: string;
		title: string;
		createdAt: number;
		updatedAt: number;
		messages: ChatTurn[];
		statusLines: StatusEntry[];
		lastResult: PipelineResult | null;
		requestError: string | null;
	};

	const sampleQuestions = [
		'Can I take ibuprofen with lisinopril for my blood pressure?',
		'Is it safe to combine warfarin and aspirin?',
		"I'm John Smith (john@example.com). Does metformin interact with alcohol?",
		'What happens if I take sertraline and tramadol together?',
		'Can I take omeprazole while on clopidogrel?',
		'Does rifampin reduce the effectiveness of warfarin?'
	];

	let msgSeq = 0;
	let chatSeq = 0;
	let statusSeq = 0;

	function nextMsgId() {
		msgSeq += 1;
		return `msg-${msgSeq}`;
	}

	function nextChatId() {
		chatSeq += 1;
		return `chat-${chatSeq}`;
	}

	function nextStatusId() {
		statusSeq += 1;
		return `status-${statusSeq}`;
	}

	function createChatSession(): ChatSession {
		const now = Date.now();
		return {
			id: nextChatId(),
			title: 'New chat',
			createdAt: now,
			updatedAt: now,
			messages: [],
			statusLines: [],
			lastResult: null,
			requestError: null
		};
	}

	let initialChat = createChatSession();
	let sessions = $state<ChatSession[]>([initialChat]);
	let activeChatId = $state(initialChat.id);
	let railCollapsed = $state(false);

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
	let threadEl = $state<HTMLDivElement | undefined>(undefined);
	let lastSentUserId = $state<string | null>(null);
	let streamingMsgId = $state<string | null>(null);
	let streamingPreAnswer = $state(false);

	let activeChat = $derived(sessions.find((session) => session.id === activeChatId) ?? sessions[0]);
	let currentMessages = $derived(activeChat?.messages ?? []);
	let currentStatus = $derived(activeChat?.statusLines ?? []);
	let currentResult = $derived(activeChat?.lastResult ?? null);
	let currentError = $derived(activeChat?.requestError ?? null);
	let latestStatus = $derived(currentStatus.at(-1)?.text ?? '');
	let hasMessages = $derived(currentMessages.length > 0);

	async function scrollThreadToEnd(behavior: ScrollBehavior = 'smooth') {
		await tick();
		threadEl?.scrollTo({ top: threadEl.scrollHeight, behavior });
	}

	function updateChat(chatId: string, updater: (session: ChatSession) => ChatSession) {
		sessions = sessions.map((session) => (session.id === chatId ? updater(session) : session));
	}

	function setChatMessages(chatId: string, messages: ChatTurn[]) {
		updateChat(chatId, (session) => ({ ...session, messages, updatedAt: Date.now() }));
	}

	function appendStatus(chatId: string, text: string) {
		updateChat(chatId, (session) => ({
			...session,
			statusLines: [...session.statusLines, { id: nextStatusId(), text }],
			updatedAt: Date.now()
		}));
	}

	function statusEntries(lines: string[]): StatusEntry[] {
		return lines.map((text) => ({ id: nextStatusId(), text }));
	}

	function titleFromQuestion(question: string) {
		const clean = question.replace(/\s+/g, ' ').trim();
		return clean.length > 42 ? `${clean.slice(0, 39)}...` : clean || 'New chat';
	}

	function esc(value: string): string {
		return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
	}

	function inlineMarkdown(text: string): string {
		return esc(text)
			.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
			.replace(/_(.+?)_/g, '<em>$1</em>')
			.replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>')
			.replace(/\n/g, '<br>');
	}

	function renderMarkdown(text: string): string {
		const paragraphs = text.split(/\n\n+/).filter((paragraph) => paragraph.trim());
		if (!paragraphs.length) return esc(text);

		return paragraphs
			.map((paragraph) => {
				const lines = paragraph.trim().split('\n');
				const firstLine = lines[0].trim();
				const headerMatch = firstLine.match(/^\*\*(.+?)\*\*$/);
				if (headerMatch) {
					const body = lines.slice(1).join('\n').trim();
					return `<section class="md-section"><h4>${esc(headerMatch[1])}</h4>${body ? `<p>${inlineMarkdown(body)}</p>` : ''}</section>`;
				}
				return `<p>${inlineMarkdown(paragraph.trim())}</p>`;
			})
			.join('');
	}

	function messageClass(message: ChatTurn) {
		return [
			'message',
			message.role,
			message.id === lastSentUserId && 'sent-flash',
			message.id === streamingMsgId && 'streaming',
			message.id === streamingMsgId && streamingPreAnswer && 'pre-answer'
		]
			.filter(Boolean)
			.join(' ');
	}

	function previewText(session: ChatSession) {
		return session.messages.at(-1)?.content ?? 'No messages yet';
	}

	function formatTime(value: number) {
		return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(value);
	}

	function formatScore(value: number | null | undefined) {
		if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
		return value.toFixed(3);
	}

	function formatMs(value: number) {
		return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${value.toFixed(0)}ms`;
	}

	function metadataLabel(source: Source) {
		const metadata = source.metadata ?? {};
		const label =
			metadata.drug ??
			metadata.title ??
			metadata.source ??
			metadata.source_name ??
			metadata.url ??
			metadata.file ??
			source.id;
		return String(label);
	}

	function metadataValue(value: MetadataValue) {
		if (value === null || value === undefined) return '';
		return String(value);
	}

	function entityTypes(result: PipelineResult) {
		return [...new Set(result.redaction.entities.map((entity) => entity.entity_type))].join(', ');
	}

	async function loadMeta() {
		try {
			const [stats, config] = await Promise.all([
				fetch('/api/corpus/stats').then((response) => response.json()),
				fetch('/api/config').then((response) => response.json())
			]);
			corpus = stats;
			apiConfig = config;
			if (!config.has_openrouter_key) skipGeneration = true;
		} catch {
			corpus = null;
		}
	}

	$effect(() => {
		void loadMeta();
	});

	function startNewChat() {
		const session = createChatSession();
		sessions = [session, ...sessions];
		activeChatId = session.id;
		input = '';
		void scrollThreadToEnd('auto');
	}

	function selectChat(chatId: string) {
		activeChatId = chatId;
		void scrollThreadToEnd('auto');
	}

	function clearActiveChat() {
		updateChat(activeChatId, (session) => ({
			...session,
			title: 'New chat',
			messages: [],
			statusLines: [],
			lastResult: null,
			requestError: null,
			updatedAt: Date.now()
		}));
		input = '';
	}

	function buildHistory(messages: ChatTurn[]): ChatTurn[] {
		const history: ChatTurn[] = [];
		for (let index = 0; index + 1 < messages.length; index += 2) {
			const user = messages[index];
			const assistant = messages[index + 1];
			if (user?.role === 'user' && assistant?.role === 'assistant') {
				history.push(user, assistant);
			}
		}
		return history;
	}

	async function send(question: string) {
		const q = question.trim();
		if (!q || loading || !activeChat) return;

		const chatId = activeChat.id;
		const previousMessages = [...activeChat.messages];
		const userMessage: ChatTurn = { id: nextMsgId(), role: 'user', content: q };
		const nextTitle = previousMessages.length ? activeChat.title : titleFromQuestion(q);

		updateChat(chatId, (session) => ({
			...session,
			title: nextTitle,
			messages: [...previousMessages, userMessage],
			statusLines: [],
			lastResult: null,
			requestError: null,
			updatedAt: Date.now()
		}));

		lastSentUserId = userMessage.id;
		setTimeout(() => {
			lastSentUserId = null;
		}, 700);

		input = '';
		loading = true;
		streamingPreAnswer = false;
		void scrollThreadToEnd();

		const history = buildHistory(previousMessages);
		let assistantId: string | null = null;
		let streamedContent = '';
		let hadPreAnswer = false;

		function ensureAssistantBubble() {
			if (assistantId !== null) return;
			assistantId = nextMsgId();
			streamingMsgId = assistantId;
			setChatMessages(chatId, [
				...previousMessages,
				userMessage,
				{ id: assistantId, role: 'assistant', content: '' }
			]);
		}

		function updateAssistantContent(content: string) {
			if (assistantId === null) return;
			updateChat(chatId, (session) => ({
				...session,
				messages: session.messages.map((message) =>
					message.id === assistantId ? { ...message, content } : message
				),
				updatedAt: Date.now()
			}));
		}

		function removeAssistantBubble() {
			if (assistantId === null) return;
			updateChat(chatId, (session) => ({
				...session,
				messages: session.messages.filter((message) => message.id !== assistantId),
				updatedAt: Date.now()
			}));
			assistantId = null;
		}

		function handleSSEEvent(event: string, rawData: string) {
			try {
				const data = JSON.parse(rawData);

				if (event === 'status') {
					appendStatus(chatId, typeof data === 'string' ? data : String(data));
				} else if (event === 'pre_answer') {
					ensureAssistantBubble();
					streamedContent = typeof data === 'string' ? data : String(data);
					hadPreAnswer = true;
					streamingPreAnswer = true;
					updateAssistantContent(streamedContent);
					if (activeChatId === chatId) void scrollThreadToEnd('auto');
				} else if (event === 'token') {
					ensureAssistantBubble();
					if (hadPreAnswer) {
						streamedContent = '';
						hadPreAnswer = false;
						streamingPreAnswer = false;
					}
					streamedContent += typeof data === 'string' ? data : String(data);
					updateAssistantContent(streamedContent);
					if (activeChatId === chatId) void scrollThreadToEnd('auto');
				} else if (event === 'result') {
					const result = data as PipelineResult;
					updateChat(chatId, (session) => ({
						...session,
						lastResult: result,
						statusLines: session.statusLines.length
							? session.statusLines
							: statusEntries(result.status_log ?? []),
						requestError: result.error,
						updatedAt: Date.now()
					}));

					if (!streamedContent) {
						ensureAssistantBubble();
						const answer =
							result.generation?.answer ??
							(result.error ? `_(generation failed: ${result.error})_` : '_(LLM call skipped - retrieval only)_');
						streamedContent = answer;
						updateAssistantContent(answer);
					}

					void loadMeta();
					if (activeChatId === chatId) void scrollThreadToEnd('auto');
				} else if (event === 'error') {
					const message = typeof data === 'string' ? data : JSON.stringify(data);
					updateChat(chatId, (session) => ({ ...session, requestError: message, updatedAt: Date.now() }));
					if (!streamedContent) removeAssistantBubble();
				}
			} catch {
				// Malformed stream frames are ignored so one bad event does not break the response.
			}
		}

		try {
			const response = await fetch('/api/chat/stream', {
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

			if (!response.ok || !response.body) {
				const errData = await response.json().catch(() => ({ detail: response.statusText }));
				throw new Error(typeof errData.detail === 'string' ? errData.detail : response.statusText);
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });

				const parts = buffer.split('\n\n');
				buffer = parts.pop() ?? '';

				for (const part of parts) {
					if (!part.trim()) continue;

					let event = '';
					const dataLines: string[] = [];
					for (const line of part.split('\n')) {
						if (line.startsWith('event: ')) event = line.slice(7).trim();
						else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
					}

					if (event && dataLines.length) handleSSEEvent(event, dataLines.join('\n'));
				}
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Request failed';
			updateChat(chatId, (session) => ({ ...session, requestError: message, updatedAt: Date.now() }));
			if (!streamedContent) removeAssistantBubble();
		} finally {
			loading = false;
			streamingMsgId = null;
			streamingPreAnswer = false;
		}
	}

	function onSubmit(event: SubmitEvent) {
		event.preventDefault();
		void send(input);
	}

	function handleComposerKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			void send(input);
		}
	}
</script>

<svelte:head>
	<title>Drug RAG Chat</title>
	<meta
		name="description"
		content="Grounded drug interaction Q&A with PII redaction, RxNorm detection, on-the-fly ingest, and citations."
	/>
</svelte:head>

<div class={['app-shell', railCollapsed && 'rail-minimized'].filter(Boolean).join(' ')}>
	<aside class="session-rail" aria-label="Current session chats">
		<div class="rail-header">
			<div class="brand-mark" aria-hidden="true">DR</div>
			{#if !railCollapsed}
				<div>
					<p class="eyebrow">Drug RAG</p>
					<h1>Session</h1>
				</div>
			{/if}
		</div>

		<div class="rail-actions">
			<button type="button" class="primary small" onclick={startNewChat} aria-label="Start a new chat">
				<span aria-hidden="true">+</span>
				{#if !railCollapsed}<span>New chat</span>{/if}
			</button>
			<button
				type="button"
				class="icon-button"
				onclick={() => (railCollapsed = !railCollapsed)}
				aria-label={railCollapsed ? 'Expand chat history' : 'Collapse chat history'}
				aria-pressed={railCollapsed}
			>
				{railCollapsed ? '>' : '<'}
			</button>
		</div>

		{#if !railCollapsed}
			<nav class="chat-list" aria-label="Chats in this session">
				{#each sessions as session (session.id)}
					<button
						type="button"
						class={['chat-tab', session.id === activeChatId && 'active'].filter(Boolean).join(' ')}
						onclick={() => selectChat(session.id)}
						aria-current={session.id === activeChatId ? 'page' : undefined}
					>
						<span class="chat-title">{session.title}</span>
						<span class="chat-preview">{previewText(session)}</span>
						<span class="chat-time">{formatTime(session.updatedAt)}</span>
					</button>
				{/each}
			</nav>
		{/if}
	</aside>

	<main class="chat-main">
		<header class="topbar">
			<div>
				<p class="eyebrow">Grounded interaction assistant</p>
				<h2>Ask about drug interactions</h2>
				<p class="subtle">PII is redacted before retrieval. Answers cite FDA and NIH public sources.</p>
			</div>

			<div class="topbar-actions">
				{#if loading}
					<div class="live-pill" role="status" aria-live="polite">
						<span class="spinner" aria-hidden="true"></span>
						<span>{latestStatus || 'Running pipeline...'}</span>
					</div>
				{/if}
				<button type="button" class="ghost small" onclick={clearActiveChat} disabled={!hasMessages && !currentResult}>
					Clear
				</button>
			</div>
		</header>

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

		{#if currentError}
			<div class="banner error" role="alert">{currentError}</div>
		{/if}

		<div class="thread" bind:this={threadEl}>
			{#if !hasMessages}
				<section class="empty-state">
					<div class="empty-orb" aria-hidden="true">Rx</div>
					<h2>Start a grounded Drug RAG chat</h2>
					<p>
						Ask about combinations, contraindications, alcohol warnings, or how a drug may affect another. The
						pipeline will redact PII, detect drug names, retrieve sources, rerank citations, and stream the answer.
					</p>
					<div class="sample-grid" aria-label="Sample prompts">
						{#each sampleQuestions.slice(0, 4) as sample (sample)}
							<button type="button" class="sample-card" disabled={loading} onclick={() => send(sample)}>
								{sample}
							</button>
						{/each}
					</div>
				</section>
			{:else}
				{#each currentMessages as message (message.id)}
					<article class={messageClass(message)}>
						<div class="avatar" aria-hidden="true">{message.role === 'user' ? 'You' : 'AI'}</div>
						<div class="message-content">
							<div class="message-label">{message.role === 'user' ? 'You' : 'Drug RAG'}</div>
							{#if message.role === 'assistant'}
								<div class="answer-body">{@html renderMarkdown(message.content)}</div>
							{:else}
								<p>{message.content}</p>
							{/if}
						</div>
					</article>
				{/each}

				{#if loading && !streamingMsgId}
					<article class="message assistant pending" aria-busy="true">
						<div class="avatar" aria-hidden="true">AI</div>
						<div class="message-content">
							<div class="message-label">Drug RAG</div>
							<p class="typing">Searching sources<span aria-hidden="true">...</span></p>
						</div>
					</article>
				{/if}
			{/if}
		</div>

		{#if currentStatus.length || currentResult}
			<section class="details-stack" aria-label="Pipeline details">
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

					<details class="detail-panel" open>
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
			</section>
		{/if}

		<form class="composer" onsubmit={onSubmit}>
			<div class="composer-main">
				<label class="sr-only" for="question-input">Ask a drug interaction question</label>
				<textarea
					id="question-input"
					bind:value={input}
					onkeydown={handleComposerKeydown}
					placeholder="Ask about a drug combination..."
					rows="2"
					disabled={loading}
				></textarea>
				<button type="submit" class="primary send" disabled={loading || !input.trim()}>
					{loading ? 'Running' : 'Send'}
				</button>
			</div>

			<div class="composer-footer">
				<details class="composer-drawer">
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
							<input
								type="checkbox"
								bind:checked={skipGeneration}
								disabled={apiConfig?.has_openrouter_key === false}
							/>
							<span>Skip LLM</span>
						</label>
					</div>
				</details>

				<details class="composer-drawer samples-drawer">
					<summary>Samples</summary>
					<div class="sample-strip">
						{#each sampleQuestions as sample (sample)}
							<button type="button" disabled={loading} onclick={() => send(sample)}>{sample}</button>
						{/each}
					</div>
				</details>
			</div>
		</form>

		<p class="disclaimer">
			Educational demo only. This is not medical advice; confirm decisions with a clinician or pharmacist.
		</p>
	</main>
</div>

<style>
	:global(body) {
		overflow: hidden;
	}

	.app-shell {
		min-height: 100vh;
		display: grid;
		grid-template-columns: 18rem minmax(0, 1fr);
		background:
			radial-gradient(circle at top left, rgba(94, 234, 212, 0.14), transparent 34rem),
			radial-gradient(circle at bottom right, rgba(96, 165, 250, 0.12), transparent 30rem),
			#070a12;
		color: #eef4ff;
	}

	.app-shell.rail-minimized {
		grid-template-columns: 5.25rem minmax(0, 1fr);
	}

	.session-rail {
		border-right: 1px solid rgba(148, 163, 184, 0.16);
		background: rgba(9, 14, 26, 0.78);
		backdrop-filter: blur(18px);
		padding: 1rem;
		display: flex;
		flex-direction: column;
		gap: 1rem;
		min-width: 0;
	}

	.rail-header,
	.rail-actions,
	.topbar,
	.topbar-actions,
	.composer-main,
	.composer-footer,
	.source-head {
		display: flex;
		align-items: center;
	}

	.rail-header {
		gap: 0.75rem;
		min-height: 3rem;
	}

	.brand-mark,
	.avatar,
	.empty-orb {
		display: grid;
		place-items: center;
		border-radius: 999px;
		background: linear-gradient(135deg, rgba(94, 234, 212, 0.22), rgba(96, 165, 250, 0.18));
		border: 1px solid rgba(148, 163, 184, 0.22);
		color: #b7fff1;
		font-weight: 800;
	}

	.brand-mark {
		width: 2.75rem;
		height: 2.75rem;
		flex: 0 0 auto;
	}

	.eyebrow {
		margin: 0;
		color: #5eead4;
		text-transform: uppercase;
		letter-spacing: 0.12em;
		font-size: 0.72rem;
		font-weight: 800;
	}

	h1,
	h2,
	h3,
	p {
		margin-top: 0;
	}

	.rail-header h1,
	.topbar h2 {
		margin: 0;
	}

	.rail-actions {
		gap: 0.5rem;
	}

	button,
	summary {
		font: inherit;
	}

	button {
		border: 0;
		cursor: pointer;
	}

	button:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	.primary,
	.ghost,
	.icon-button,
	.chat-tab,
	.sample-card,
	.sample-strip button {
		border-radius: 999px;
		transition:
			transform 0.16s ease,
			border-color 0.16s ease,
			background 0.16s ease;
	}

	.primary {
		background: #5eead4;
		color: #03110f;
		font-weight: 800;
		padding: 0.8rem 1.05rem;
	}

	.primary:hover:not(:disabled),
	.ghost:hover:not(:disabled),
	.icon-button:hover,
	.chat-tab:hover,
	.sample-card:hover:not(:disabled),
	.sample-strip button:hover:not(:disabled) {
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

	.icon-button,
	.ghost {
		border: 1px solid rgba(148, 163, 184, 0.18);
		background: rgba(15, 23, 42, 0.72);
		color: #e2e8f0;
	}

	.icon-button {
		width: 2.45rem;
		height: 2.45rem;
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
		color: #dbeafe;
		background: rgba(15, 23, 42, 0.58);
		border: 1px solid rgba(148, 163, 184, 0.12);
	}

	.chat-tab.active {
		background: rgba(20, 184, 166, 0.14);
		border-color: rgba(94, 234, 212, 0.42);
	}

	.chat-title,
	.chat-preview {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.chat-title {
		font-weight: 800;
	}

	.chat-preview,
	.chat-time,
	.subtle,
	.muted,
	.disclaimer {
		color: #94a3b8;
	}

	.chat-preview,
	.chat-time {
		font-size: 0.8rem;
	}

	.chat-main {
		height: 100vh;
		min-width: 0;
		display: grid;
		grid-template-rows: auto auto auto minmax(0, 1fr) auto auto;
		gap: 0.85rem;
		padding: 1.1rem;
	}

	.topbar {
		justify-content: space-between;
		gap: 1rem;
	}

	.topbar h2 {
		font-size: clamp(1.45rem, 2vw, 2.2rem);
	}

	.topbar .subtle {
		margin: 0.25rem 0 0;
	}

	.topbar-actions {
		gap: 0.65rem;
		flex-wrap: wrap;
		justify-content: flex-end;
	}

	.live-pill {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.55rem 0.75rem;
		border-radius: 999px;
		background: rgba(94, 234, 212, 0.12);
		color: #c8fff6;
		border: 1px solid rgba(94, 234, 212, 0.24);
		max-width: min(32rem, 52vw);
	}

	.live-pill span:last-child {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.spinner {
		width: 0.9rem;
		height: 0.9rem;
		border-radius: 999px;
		border: 2px solid rgba(94, 234, 212, 0.3);
		border-top-color: #5eead4;
		animation: spin 0.8s linear infinite;
		flex: 0 0 auto;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.meta-row {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.8rem;
	}

	.meta-card,
	.detail-panel,
	.composer {
		border: 1px solid rgba(148, 163, 184, 0.16);
		background: rgba(15, 23, 42, 0.68);
		box-shadow: 0 18px 60px rgba(0, 0, 0, 0.24);
		backdrop-filter: blur(18px);
	}

	.meta-card,
	.detail-panel {
		border-radius: 1.1rem;
		padding: 0.9rem;
	}

	summary {
		cursor: pointer;
		font-weight: 800;
		color: #f8fafc;
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
		background: rgba(2, 6, 23, 0.35);
		border: 1px solid rgba(148, 163, 184, 0.1);
		display: grid;
		gap: 0.2rem;
	}

	.metric-grid strong {
		overflow-wrap: anywhere;
	}

	.metric-grid span {
		color: #94a3b8;
		font-size: 0.78rem;
	}

	.mono,
	pre {
		font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
	}

	.banner {
		border-radius: 0.9rem;
		padding: 0.8rem 1rem;
	}

	.banner.error {
		background: rgba(251, 113, 133, 0.12);
		border: 1px solid rgba(251, 113, 133, 0.3);
		color: #fecdd3;
	}

	.thread {
		min-height: 0;
		overflow: auto;
		padding: 0.4rem 0.25rem 1rem;
		scroll-behavior: smooth;
	}

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
		width: 4.2rem;
		height: 4.2rem;
	}

	.empty-state h2 {
		margin: 0;
		font-size: clamp(1.8rem, 4vw, 3.4rem);
		letter-spacing: -0.05em;
	}

	.empty-state p {
		max-width: 43rem;
		color: #b6c2d4;
	}

	.sample-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.75rem;
		width: min(100%, 46rem);
	}

	.sample-card,
	.sample-strip button {
		color: #dbeafe;
		background: rgba(15, 23, 42, 0.72);
		border: 1px solid rgba(148, 163, 184, 0.16);
	}

	.sample-card {
		border-radius: 1rem;
		padding: 0.9rem;
		text-align: left;
	}

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
		width: 2.5rem;
		height: 2.5rem;
		font-size: 0.78rem;
	}

	.message.user .avatar {
		background: rgba(96, 165, 250, 0.2);
		color: #bfdbfe;
	}

	.message-content {
		border-radius: 1.2rem;
		padding: 0.9rem 1rem;
		background: rgba(15, 23, 42, 0.78);
		border: 1px solid rgba(148, 163, 184, 0.14);
	}

	.message.user .message-content {
		background: rgba(37, 99, 235, 0.18);
		border-color: rgba(96, 165, 250, 0.26);
	}

	.message.pre-answer .message-content {
		border-color: rgba(251, 191, 36, 0.32);
	}

	.message-label {
		color: #94a3b8;
		font-size: 0.78rem;
		font-weight: 800;
		margin-bottom: 0.35rem;
	}

	.message-content p {
		margin: 0;
		white-space: pre-wrap;
	}

	.answer-body :global(p) {
		margin: 0 0 0.85rem;
	}

	.answer-body :global(p:last-child) {
		margin-bottom: 0;
	}

	.answer-body :global(.cite) {
		color: #5eead4;
		font-weight: 800;
	}

	.answer-body :global(.md-section h4) {
		margin: 0 0 0.35rem;
		color: #e0f2fe;
	}

	.typing {
		color: #b6c2d4;
	}

	.sent-flash .message-content {
		animation: flash 0.7s ease;
	}

	@keyframes flash {
		0% {
			box-shadow: 0 0 0 0 rgba(94, 234, 212, 0.35);
		}
		100% {
			box-shadow: 0 0 0 14px rgba(94, 234, 212, 0);
		}
	}

	.details-stack {
		display: grid;
		gap: 0.65rem;
		max-height: 34vh;
		overflow: auto;
		padding-right: 0.2rem;
	}

	.detail-panel {
		padding: 0.85rem 1rem;
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
		color: #dbeafe;
		background: rgba(2, 6, 23, 0.5);
		border: 1px solid rgba(148, 163, 184, 0.12);
	}

	.note {
		border-radius: 0.8rem;
		padding: 0.7rem;
		margin-bottom: 0;
	}

	.ok,
	.ok-text {
		color: #86efac;
	}

	.warn,
	.warn-text {
		color: #fcd34d;
	}

	.pill-list,
	.metadata,
	.sample-strip {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.data-pill,
	.metadata span {
		border-radius: 999px;
		border: 1px solid rgba(148, 163, 184, 0.14);
		background: rgba(2, 6, 23, 0.36);
		padding: 0.45rem 0.65rem;
	}

	.data-pill {
		display: grid;
		gap: 0.1rem;
	}

	.data-pill small {
		color: #94a3b8;
	}

	.ingest-box,
	.source-card {
		border-radius: 0.95rem;
		padding: 0.85rem;
		background: rgba(2, 6, 23, 0.34);
		border: 1px solid rgba(148, 163, 184, 0.12);
		margin-top: 0.8rem;
	}

	.source-list {
		display: grid;
		gap: 0.7rem;
	}

	.source-head {
		justify-content: space-between;
		gap: 1rem;
	}

	.score {
		color: #5eead4;
		font-size: 0.82rem;
	}

	.metadata {
		margin-top: 0.65rem;
	}

	.metadata span {
		color: #b6c2d4;
		font-size: 0.78rem;
	}

	.status-list {
		margin-bottom: 0;
		color: #cbd5e1;
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
		background: rgba(2, 6, 23, 0.36);
	}

	.composer {
		border-radius: 1.35rem;
		padding: 0.75rem;
	}

	.composer-main {
		gap: 0.75rem;
	}

	textarea {
		flex: 1;
		min-width: 0;
		resize: none;
		border: 0;
		outline: none;
		border-radius: 1rem;
		padding: 0.9rem 1rem;
		background: rgba(2, 6, 23, 0.5);
		color: #f8fafc;
		font: inherit;
		line-height: 1.45;
	}

	textarea::placeholder {
		color: #64748b;
	}

	.send {
		align-self: stretch;
		min-width: 6.5rem;
	}

	.composer-footer {
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
		background: rgba(15, 23, 42, 0.8);
		border: 1px solid rgba(148, 163, 184, 0.16);
	}

	.composer-drawer summary::-webkit-details-marker {
		display: none;
	}

	.composer-drawer[open] summary {
		border-color: rgba(94, 234, 212, 0.38);
	}

	.control-grid,
	.sample-strip {
		position: absolute;
		bottom: calc(100% + 0.6rem);
		left: 0;
		z-index: 10;
		width: min(28rem, 82vw);
		border-radius: 1rem;
		border: 1px solid rgba(148, 163, 184, 0.18);
		background: rgba(8, 13, 24, 0.96);
		box-shadow: 0 18px 60px rgba(0, 0, 0, 0.42);
		padding: 0.85rem;
	}

	.control-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.85rem;
	}

	.control-grid label {
		display: grid;
		gap: 0.35rem;
		color: #dbeafe;
	}

	.control-grid output {
		color: #5eead4;
		font-weight: 800;
	}

	input[type='range'] {
		accent-color: #5eead4;
	}

	input[type='checkbox'] {
		width: 1rem;
		height: 1rem;
		accent-color: #5eead4;
	}

	.check-row {
		grid-template-columns: auto 1fr;
		align-items: center;
	}

	.samples-drawer .sample-strip {
		width: min(42rem, 88vw);
	}

	.sample-strip button {
		padding: 0.55rem 0.75rem;
		text-align: left;
	}

	.disclaimer {
		margin: 0;
		font-size: 0.82rem;
		text-align: center;
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
		.app-shell,
		.app-shell.rail-minimized {
			grid-template-columns: 1fr;
		}

		.session-rail {
			position: fixed;
			inset: auto 0 0 0;
			z-index: 20;
			max-height: 42vh;
			border-right: 0;
			border-top: 1px solid rgba(148, 163, 184, 0.16);
		}

		.app-shell.rail-minimized .session-rail {
			left: auto;
			width: 5rem;
			border-left: 1px solid rgba(148, 163, 184, 0.16);
		}

		.chat-main {
			padding-bottom: 6rem;
		}

		.meta-row,
		.sample-grid,
		.split,
		.control-grid {
			grid-template-columns: 1fr;
		}

		.topbar {
			align-items: flex-start;
			flex-direction: column;
		}

		.live-pill {
			max-width: 100%;
		}
	}

	@media (max-width: 640px) {
		.chat-main {
			padding: 0.75rem 0.75rem 6rem;
		}

		.composer-main {
			align-items: stretch;
			flex-direction: column;
		}

		.send {
			min-height: 2.8rem;
		}

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
