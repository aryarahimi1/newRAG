<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { browser } from '$app/environment';
	import DOMPurify from 'isomorphic-dompurify';

	import { applyTheme, resolveInitialTheme, type Theme } from '$lib/theme';

	const SESSIONS_STORAGE_KEY = 'medication-reference-chat-sessions-v1';
	const DEV_MODE_STORAGE_KEY = 'medication-reference-dev-mode';

	type Role = 'user' | 'assistant';
	type RecallSeverity = 'class1' | 'class2' | 'class3' | 'unknown';
	type RecallAlert = {
		id: string;
		drugName: string;
		severity: RecallSeverity;
		label: string;
		sourceUrl: string;
	};
	type ChatTurn = { id: string; role: Role; content: string; recallAlerts?: RecallAlert[] };
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
			title: 'Empty conversation',
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
	/** After true, session writes to localStorage are allowed (avoids clobbering before restore). */
	let sessionStorageReady = $state(false);
	/** Default closed so mobile CSS (drawer) does not flash open before hydration. Desktop is expanded in onMount. */
	let railCollapsed = $state(true);

	let isMobile = $state(false);
	/** Last mqMobile.matches — used to close the rail when switching from desktop → mobile. */
	let wasMobile = false;
	let controlsOpen = $state(false);

	/** Pipeline debug UI, corpus/model panels, and retrieval sliders — never shown to end users. */
	let devModeEnabled = $state(false);

	let showMobileBackdrop = $derived(isMobile && (!railCollapsed || (devModeEnabled && controlsOpen)));

	function dismissMobileOverlays() {
		railCollapsed = true;
		controlsOpen = false;
	}

	function toggleDevMode() {
		if (!browser) return;
		try {
			if (sessionStorage.getItem(DEV_MODE_STORAGE_KEY) === '1') {
				sessionStorage.removeItem(DEV_MODE_STORAGE_KEY);
				devModeEnabled = false;
			} else {
				sessionStorage.setItem(DEV_MODE_STORAGE_KEY, '1');
				devModeEnabled = true;
			}
		} catch {
			devModeEnabled = !devModeEnabled;
		}
	}

	function toggleTheme() {
		theme = theme === 'light' ? 'dark' : 'light';
		applyTheme(theme);
	}

	function toggleRailCollapsed() {
		if (railCollapsed) {
			railCollapsed = false;
			if (isMobile) {
				controlsOpen = false;
			}
		} else {
			railCollapsed = true;
		}
	}

	function openMobileSessionRail() {
		railCollapsed = false;
		if (isMobile) controlsOpen = false;
	}

	function onControlsToggle(e: ToggleEvent & { currentTarget: HTMLDetailsElement }) {
		if (!isMobile || !e.currentTarget.open) return;
		railCollapsed = true;
	}

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
	/** Narrow viewport: pipeline block starts collapsed so the message thread keeps height. */
	let compactPipelineUi = $state(false);
	/** false until matchMedia runs so mobile first paint is not an expanded 34vh block */
	let pipelineDetailsOpen = $state(false);
	let threadEl = $state<HTMLDivElement | undefined>(undefined);
	let lastSentUserId = $state<string | null>(null);
	let streamingMsgId = $state<string | null>(null);
	let streamingPreAnswer = $state(false);

	let theme = $state<Theme>('light');

	let activeChat = $derived(sessions.find((session) => session.id === activeChatId) ?? sessions[0]);
	let currentMessages = $derived(activeChat?.messages ?? []);
	let hasAssistantReply = $derived(currentMessages.some((m) => m.role === 'assistant'));
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

	function firstUserContent(session: ChatSession): string | undefined {
		const turn = session.messages.find((m) => m.role === 'user');
		const t = turn?.content?.replace(/\s+/g, ' ').trim();
		return t || undefined;
	}

	/** Sidebar primary line: always follows the first user question when present. */
	function chatListTitle(session: ChatSession): string {
		const first = firstUserContent(session);
		if (first) return titleFromQuestion(first);
		return 'Empty conversation';
	}

	/** Sidebar secondary line: status / last activity snippet. */
	function chatListPreview(session: ChatSession): string {
		const msgs = session.messages;
		if (msgs.length === 0) return 'Ask your first question below';
		const last = msgs.at(-1)!;
		if (last.role === 'user' && msgs.length === 1) return 'Awaiting response…';
		const raw = last.content.replace(/\s+/g, ' ').trim() || '…';
		return raw.length > 52 ? `${raw.slice(0, 49)}…` : raw;
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

	/** Safe HTML for {@html}: markdown pipeline output is sanitized (escaping alone is insufficient once tags are re-injected). */
	function safeMarkdownHtml(text: string): string {
		const raw = renderMarkdown(text);
		return DOMPurify.sanitize(raw, {
			ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'span', 'section', 'h4'],
			ALLOWED_ATTR: ['class']
		});
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

	function recallSeverityFromSection(section: string): RecallSeverity {
		if (/class\s*iii\b/i.test(section)) return 'class3';
		if (/class\s*ii\b/i.test(section)) return 'class2';
		if (/class\s*i\b/i.test(section)) return 'class1';
		return 'unknown';
	}

	function recallLabelForSeverity(sev: RecallSeverity): string {
		if (sev === 'class1') return 'Class I recall';
		if (sev === 'class2') return 'Class II recall';
		if (sev === 'class3') return 'Class III recall';
		return 'FDA recall';
	}

	/** Union retrieved + reranked so recalls dropped by reranking still surface in the UI. */
	function extractRecallAlerts(result: PipelineResult): RecallAlert[] {
		const merged = [...(result.retrieved ?? []), ...(result.reranked ?? [])];
		const seen = new Set<string>();
		const list: RecallAlert[] = [];
		for (const src of merged) {
			const meta = src.metadata ?? {};
			if (String(meta.source ?? '') !== 'openfda_recall') continue;
			if (seen.has(src.id)) continue;
			const section = String(meta.section ?? '');
			const sev = recallSeverityFromSection(section);
			const drugName = String(meta.drug_name ?? 'unknown');
			const url = String(meta.source_url ?? '');
			if (!url.startsWith('http')) continue;
			const label = recallLabelForSeverity(sev);
			seen.add(src.id);
			list.push({
				id: src.id,
				drugName,
				severity: sev,
				label,
				sourceUrl: url
			});
		}
		list.sort((a, b) => {
			const rank = (s: RecallSeverity) =>
				s === 'class1' ? 0 : s === 'class2' ? 1 : s === 'class3' ? 2 : 3;
			const d = rank(a.severity) - rank(b.severity);
			if (d !== 0) return d;
			return a.drugName.localeCompare(b.drugName);
		});
		return list;
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

	$effect(() => {
		if (!devModeEnabled) controlsOpen = false;
	});

	onMount(() => {
		if (!browser) return;
		if (!window.matchMedia('(max-width: 920px)').matches) {
			railCollapsed = false;
		}
		theme = resolveInitialTheme();
		applyTheme(theme);

		try {
			const params = new URLSearchParams(window.location.search);
			if (params.get('dev') === 'true') {
				sessionStorage.setItem(DEV_MODE_STORAGE_KEY, '1');
				params.delete('dev');
				const qs = params.toString();
				const next = qs ? `${window.location.pathname}?${qs}${window.location.hash}` : `${window.location.pathname}${window.location.hash}`;
				history.replaceState({}, '', next);
			}
			devModeEnabled = sessionStorage.getItem(DEV_MODE_STORAGE_KEY) === '1';
		} catch {
			devModeEnabled = false;
		}
		try {
			const raw = localStorage.getItem(SESSIONS_STORAGE_KEY);
			if (raw) {
				const data = JSON.parse(raw) as {
					sessions?: ChatSession[];
					activeChatId?: string;
					msgSeq?: number;
					chatSeq?: number;
					statusSeq?: number;
				};
				if (Array.isArray(data.sessions) && data.sessions.length > 0) {
					sessions = data.sessions;
					if (data.activeChatId && data.sessions.some((s) => s.id === data.activeChatId)) {
						activeChatId = data.activeChatId;
					} else {
						activeChatId = data.sessions[0].id;
					}
					if (typeof data.msgSeq === 'number') msgSeq = data.msgSeq;
					if (typeof data.chatSeq === 'number') chatSeq = data.chatSeq;
					if (typeof data.statusSeq === 'number') statusSeq = data.statusSeq;
				}
			}
		} catch {
			// ignore corrupt storage
		} finally {
			sessionStorageReady = true;
		}
	});

	$effect(() => {
		if (!browser || !sessionStorageReady) return;
		void sessions;
		void activeChatId;
		void msgSeq;
		void chatSeq;
		void statusSeq;
		try {
			localStorage.setItem(
				SESSIONS_STORAGE_KEY,
				JSON.stringify({ sessions, activeChatId, msgSeq, chatSeq, statusSeq })
			);
		} catch {
			// quota / private mode
		}
	});

	$effect(() => {
		const mqMobile = window.matchMedia('(max-width: 920px)');
		const mqCompact = window.matchMedia('(max-width: 640px)');
		const sync = () => {
			const nextMobile = mqMobile.matches;
			if (nextMobile && !wasMobile) {
				railCollapsed = true;
			} else if (!nextMobile && wasMobile) {
				railCollapsed = false;
			}
			wasMobile = nextMobile;
			isMobile = nextMobile;
			compactPipelineUi = mqCompact.matches;
			pipelineDetailsOpen = !mqCompact.matches;
		};
		sync();
		mqMobile.addEventListener('change', sync);
		mqCompact.addEventListener('change', sync);
		return () => {
			mqMobile.removeEventListener('change', sync);
			mqCompact.removeEventListener('change', sync);
		};
	});

	$effect(() => {
		if (!browser) return;
		const lockScroll = isMobile && !railCollapsed;
		document.body.style.overflow = lockScroll ? 'hidden' : '';
		return () => {
			document.body.style.overflow = '';
		};
	});

	function startNewChat() {
		const session = createChatSession();
		sessions = [session, ...sessions];
		activeChatId = session.id;
		input = '';
		if (isMobile) {
			railCollapsed = true;
		}
		void scrollThreadToEnd('auto');
	}

	function selectChat(chatId: string) {
		activeChatId = chatId;
		if (isMobile) {
			railCollapsed = true;
		}
		void scrollThreadToEnd('auto');
	}

	function clearActiveChat() {
		updateChat(activeChatId, (session) => ({
			...session,
			title: 'Empty conversation',
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
					// Ensure assistant row exists so recalls attach even when there were no streamed tokens.
					ensureAssistantBubble();
					const recalls = extractRecallAlerts(result);
					updateChat(chatId, (session) => {
						let messages = session.messages;
						if (assistantId !== null) {
							messages = session.messages.map((m) =>
								m.id === assistantId
									? {
											...m,
											recallAlerts: recalls.length ? recalls : undefined
										}
									: m
							);
						}
						return {
							...session,
							messages,
							lastResult: result,
							statusLines: session.statusLines.length
								? session.statusLines
								: statusEntries(result.status_log ?? []),
							requestError: result.error,
							updatedAt: Date.now()
						};
					});

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
	<title>Medication Reference</title>
	<meta
		name="description"
		content="Ask about drug interactions and medication safety. Personal details are protected before lookup; answers cite FDA and NIH sources."
	/>
	<meta name="theme-color" content={theme === 'dark' ? '#0b1120' : '#fafafa'} />
</svelte:head>

<svelte:window
	onkeydown={(e) => {
		if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'd') {
			e.preventDefault();
			toggleDevMode();
			return;
		}
		if (e.key === 'Escape' && showMobileBackdrop) {
			e.preventDefault();
			dismissMobileOverlays();
		}
	}}
/>

<div class={['app-shell', railCollapsed && 'rail-minimized'].filter(Boolean).join(' ')}>
	{#snippet pharmaMark()}
		<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
			<text
				x="12"
				y="16.5"
				text-anchor="middle"
				fill="currentColor"
				font-size="14"
				font-family="Georgia, 'Times New Roman', 'Noto Serif', serif"
				font-weight="600"
			>℞</text>
		</svg>
	{/snippet}
	{#snippet userAvatarGlyph()}
		<svg class="avatar-glyph" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
			<path
				stroke-linecap="round"
				stroke-linejoin="round"
				d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z"
			/>
		</svg>
	{/snippet}
	{#snippet assistantAvatarGlyph()}
		<svg class="avatar-glyph" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
			<path stroke-linecap="round" d="M12 3v1.5M9 2.5v2M15 2.5v2" />
			<rect x="4.5" y="6.5" width="15" height="12" rx="2" />
			<circle cx="9.5" cy="12" r="0.9" fill="currentColor" stroke="none" />
			<circle cx="14.5" cy="12" r="0.9" fill="currentColor" stroke="none" />
			<path stroke-linecap="round" d="M9.5 16h5" />
		</svg>
	{/snippet}
	{#snippet railToggleIcon()}
		<svg class="rail-toggle-svg" viewBox="0 0 24 24" fill="none" aria-hidden="true">
			<path
				class="rail-toggle-path"
				d="M9 5l6 7-6 7"
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-linejoin="round"
			/>
		</svg>
	{/snippet}
	{#if showMobileBackdrop}
		<button
			type="button"
			class="mobile-overlay-backdrop"
			aria-label="Close panel"
			onclick={dismissMobileOverlays}
		></button>
	{/if}
	<aside
		id="session-rail-panel"
		class="session-rail"
		aria-label="Current session chats"
	>
		<div class="rail-header">
			<div class="brand-mark" aria-hidden="true">{@render pharmaMark()}</div>
			{#if !railCollapsed}
				<div>
					<p class="eyebrow">Medication Reference</p>
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
				class="icon-button rail-toggle"
				onclick={toggleRailCollapsed}
				aria-label={railCollapsed ? 'Expand chat history' : 'Collapse chat history'}
				aria-expanded={!railCollapsed}
			>
				{@render railToggleIcon()}
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
						<span class="chat-title">{chatListTitle(session)}</span>
						<span class="chat-preview">{chatListPreview(session)}</span>
						<span class="chat-time">{formatTime(session.updatedAt)}</span>
					</button>
				{/each}
			</nav>
		{/if}
	</aside>

	<main class="chat-main">
		<div class="main-max">
		<header class="topbar">
			<div class="topbar-primary">
				{#if isMobile}
					<button
						type="button"
						class="icon-button mobile-menu-btn"
						onclick={openMobileSessionRail}
						aria-label="Open chat history"
						aria-controls="session-rail-panel"
						aria-expanded={!railCollapsed}
					>
						<svg class="hamburger-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
							<path
								d="M4 6h16M4 12h16M4 18h16"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
							/>
						</svg>
					</button>
				{/if}
				<div class="topbar-titles">
					<p class="eyebrow">Citation-backed reference</p>
					<h2>Ask about drug interactions</h2>
					<p class="subtle">PII is redacted before retrieval. Answers cite FDA and NIH public sources.</p>
				</div>
			</div>

			<div class="topbar-actions">
				<button
					type="button"
					class="ghost small theme-toggle"
					onclick={toggleTheme}
					aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
					title={theme === 'light' ? 'Dark mode' : 'Light mode'}
				>
					{#if theme === 'light'}
						<svg class="theme-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
							<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" stroke-linecap="round" stroke-linejoin="round" />
						</svg>
					{:else}
						<svg class="theme-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
							<circle cx="12" cy="12" r="5" />
							<path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" stroke-linecap="round" />
						</svg>
					{/if}
				</button>
				<button type="button" class="ghost small" onclick={clearActiveChat} disabled={!hasMessages && !currentResult}>
					Clear
				</button>
			</div>
		</header>

		{#if devModeEnabled && hasAssistantReply}
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
		{/if}

		{#if currentError}
			<div class="banner error" role="alert">{currentError}</div>
		{/if}

		<div class="thread" bind:this={threadEl}>
			{#if !hasMessages}
				<section class="empty-state">
					<div class="empty-orb" aria-hidden="true">{@render pharmaMark()}</div>
					<h2>Ask your first question</h2>
					<p>
						Ask about combinations, contraindications, alcohol warnings, or how one medication may affect another.
						Personal details are removed before lookup; answers are drawn from retrieved FDA and NIH sources and shown
						with citations.
					</p>
					<div class="sample-grid" aria-label="Example questions">
						{#each sampleQuestions as sample (sample)}
							<button type="button" class="sample-card" disabled={loading} onclick={() => send(sample)}>
								{sample}
							</button>
						{/each}
					</div>
				</section>
			{:else}
				{#each currentMessages as message (message.id)}
					<article class={messageClass(message)}>
						<div class="avatar" aria-hidden="true">
							{#if message.role === 'user'}
								{@render userAvatarGlyph()}
							{:else}
								{@render assistantAvatarGlyph()}
							{/if}
						</div>
						<div class="message-content">
							<div class="message-label">{message.role === 'user' ? 'You' : 'Assistant'}</div>
							{#if message.role === 'assistant'}
								{#if message.recallAlerts?.length}
									<div
										class="recall-strip"
										role="group"
										aria-label="FDA drug recalls found in sources for this answer"
									>
										<p class="recall-strip-title">FDA recalls in sources</p>
										<div class="recall-chips">
											{#each message.recallAlerts as alert (alert.id)}
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
								{/if}
								<div class="answer-body">{@html safeMarkdownHtml(message.content)}</div>
							{:else}
								<p>{message.content}</p>
							{/if}
						</div>
					</article>
				{/each}

				{#if loading && !streamingMsgId}
					<article class="message assistant pending" aria-busy="true">
						<div class="avatar" aria-hidden="true">{@render assistantAvatarGlyph()}</div>
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

		{#if devModeEnabled && (currentStatus.length || currentResult)}
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
		{/if}

		<div class={['composer', controlsOpen && devModeEnabled && 'composer-drawer-active'].filter(Boolean).join(' ')}>
			<form class="composer-form" onsubmit={onSubmit}>
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
								<input
									type="checkbox"
									bind:checked={skipGeneration}
									disabled={apiConfig?.has_openrouter_key === false}
								/>
								<span>Skip LLM</span>
							</label>
						</div>
					</details>
				</div>
			{/if}
		</div>
		</div>
	</main>
</div>

<style>
	.app-shell {
		min-height: 100dvh;
		display: grid;
		grid-template-columns: 18rem minmax(0, 1fr);
		background: var(--shell-bg);
		color: var(--shell-fg);
	}

	.app-shell.rail-minimized {
		grid-template-columns: 5.25rem minmax(0, 1fr);
	}

	.session-rail {
		border-right: 1px solid var(--rail-border);
		background: var(--rail-bg);
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

	.brand-mark {
		display: grid;
		place-items: center;
		width: 2.75rem;
		height: 2.75rem;
		flex: 0 0 auto;
		border-radius: 10px;
		background: var(--brand-mark-bg);
		border: 1px solid var(--brand-mark-border);
		color: var(--brand-mark-fg);
		padding: 0.45rem;
		box-sizing: border-box;
	}

	.brand-mark svg {
		width: 100%;
		height: 100%;
		display: block;
	}

	.avatar,
	.empty-orb {
		display: grid;
		place-items: center;
		border-radius: 999px;
		background: var(--chip-bg);
		border: 1px solid var(--chip-border);
		color: var(--chip-fg);
		font-weight: 800;
	}

	.avatar :global(.avatar-glyph) {
		width: 1.1rem;
		height: 1.1rem;
		display: block;
	}

	.empty-orb {
		width: 4.5rem;
		height: 4.5rem;
		border-radius: 14px;
		background: var(--empty-orb-bg);
		border: 1px solid var(--empty-orb-border);
		color: var(--chip-fg);
		padding: 0.85rem;
		box-sizing: border-box;
	}

	.empty-orb svg {
		width: 100%;
		height: 100%;
		display: block;
	}

	.eyebrow {
		margin: 0;
		color: var(--eyebrow);
		text-transform: uppercase;
		letter-spacing: 0.12em;
		font-size: 0.72rem;
		font-weight: 500;
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
		color: var(--text-heading);
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
	.sample-card {
		border-radius: 999px;
		transition:
			transform 0.16s ease,
			border-color 0.16s ease,
			background 0.16s ease;
	}

	.primary {
		background: var(--btn-primary);
		color: var(--brand-mark-fg);
		font-weight: 600;
		padding: 0.8rem 1.05rem;
	}

	.primary:hover:not(:disabled) {
		background: var(--btn-primary-hover);
	}

	.primary:hover:not(:disabled),
	.ghost:hover:not(:disabled),
	.icon-button:hover,
	.chat-tab:hover,
	.sample-card:hover:not(:disabled) {
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
		border: 1px solid var(--btn-ghost-border);
		background: var(--btn-ghost-bg);
		color: var(--btn-ghost-fg);
		font-weight: 500;
	}

	.theme-toggle {
		display: inline-grid;
		place-items: center;
		min-width: 2.45rem;
		padding-left: 0.62rem;
		padding-right: 0.62rem;
	}

	.theme-icon {
		width: 1.15rem;
		height: 1.15rem;
		display: block;
	}

	.icon-button {
		width: 2.45rem;
		height: 2.45rem;
	}

	.rail-toggle {
		display: inline-grid;
		place-items: center;
		padding: 0;
	}

	.rail-toggle .rail-toggle-svg {
		width: 1.15rem;
		height: 1.15rem;
		color: var(--btn-ghost-fg);
	}

	/* Collapsed: chevron right (open rail); expanded: chevron left (minimize) */
	.rail-toggle .rail-toggle-path {
		transform-origin: 12px 12px;
		transform: scaleX(-1);
	}

	:global(.rail-minimized) .rail-toggle .rail-toggle-path {
		transform: none;
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
		border-radius: var(--radius-md);
		transition:
			transform 0.16s ease,
			border-color 0.16s ease,
			background 0.16s ease;
		color: var(--tab-fg);
		background: var(--tab-bg);
		border: 1px solid var(--border);
	}

	.chat-tab.active {
		background: var(--tab-active-bg);
		border-color: var(--tab-active-border);
	}

	.chat-title,
	.chat-preview {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.chat-title {
		font-weight: 600;
	}

	.chat-preview,
	.chat-time,
	.subtle,
	.muted {
		color: var(--text-muted);
	}

	.chat-preview,
	.chat-time {
		font-size: 0.8rem;
	}

	.chat-main {
		height: 100dvh;
		min-height: 100dvh;
		min-width: 0;
		padding: 1.1rem;
	}

	.main-max {
		max-width: 900px;
		width: 100%;
		margin: 0 auto;
		min-width: 0;
		min-height: 0;
		height: 100%;
		display: flex;
		flex-direction: column;
		gap: 0.85rem;
	}

	.topbar {
		justify-content: space-between;
		gap: 1rem;
	}

	.topbar-primary {
		display: flex;
		align-items: flex-start;
		gap: 0.65rem;
		min-width: 0;
		flex: 1;
	}

	.topbar-titles {
		min-width: 0;
	}

	.mobile-menu-btn {
		flex: 0 0 auto;
		margin-top: 0.1rem;
	}

	.hamburger-icon {
		display: block;
		width: 1.2rem;
		height: 1.2rem;
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

	.meta-row {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.8rem;
	}

	.meta-card,
	.detail-panel,
	.composer {
		border: 1px solid var(--border);
		background: var(--card-bg);
		box-shadow: var(--card-shadow);
	}

	.meta-card,
	.detail-panel {
		border-radius: 1.1rem;
		padding: 0.9rem;
	}

	summary {
		cursor: pointer;
		font-weight: 500;
		color: var(--summary-color);
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

	.mono,
	pre {
		font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
	}

	.banner {
		border-radius: 0.9rem;
		padding: 0.8rem 1rem;
	}

	.banner.error {
		background: var(--error-banner-bg);
		border: 1px solid var(--error-banner-border);
		color: var(--error-banner-fg);
	}

	.thread {
		flex: 1;
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

	.empty-state h2 {
		margin: 0;
		font-size: clamp(1.8rem, 4vw, 3.4rem);
		letter-spacing: -0.05em;
		color: var(--text-heading);
	}

	.empty-state p {
		max-width: 43rem;
		color: var(--empty-body);
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

	/* Ring matches user bubble: cool blue (same family as #eff6ff / #1e40af), not green-teal */
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
		color: var(--pre-fg);
		background: var(--pre-bg);
		border: 1px solid var(--pre-border);
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

	.composer {
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
		background: var(--composer-drawer-summary-bg);
		border: 1px solid var(--composer-drawer-summary-border);
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
		.mobile-overlay-backdrop {
			display: block;
			position: fixed;
			inset: 0;
			z-index: 35;
			margin: 0;
			padding: 0;
			border: 0;
			border-radius: 0;
			background: var(--mobile-overlay);
			backdrop-filter: blur(2px);
			cursor: pointer;
			-webkit-tap-highlight-color: transparent;
		}

		/* Single main column; rail is a fixed left drawer (not in layout flow). */
		.app-shell,
		.app-shell.rail-minimized {
			display: block;
		}

		.session-rail {
			position: fixed;
			z-index: 40;
			top: 0;
			bottom: 0;
			left: 0;
			width: min(20rem, 90vw);
			max-width: 100%;
			height: 100dvh;
			max-height: 100dvh;
			margin: 0;
			border-right: 1px solid var(--rail-border);
			border-top: none;
			box-shadow: none;
			transform: translate3d(-100%, 0, 0);
			transition: transform 0.22s ease, box-shadow 0.22s ease;
			overflow-y: auto;
			overflow-x: hidden;
			overscroll-behavior: contain;
			-webkit-overflow-scrolling: touch;
			padding: 1rem;
			padding-top: max(1rem, env(safe-area-inset-top));
			padding-left: max(1rem, env(safe-area-inset-left));
			padding-bottom: max(1rem, env(safe-area-inset-bottom));
		}

		.app-shell:not(.rail-minimized) .session-rail {
			transform: translate3d(0, 0, 0);
			box-shadow: 4px 0 32px color-mix(in srgb, var(--shell-fg) 16%, transparent);
		}

		@media (prefers-reduced-motion: reduce) {
			.session-rail {
				transition: none;
			}
		}

		.chat-main {
			height: auto;
			min-height: 100dvh;
			padding-bottom: max(1.1rem, env(safe-area-inset-bottom));
		}

		.composer.composer-drawer-active {
			position: relative;
			z-index: 45;
		}

		.composer-drawer[open] .control-grid {
			z-index: 46;
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
	}

	@media (max-width: 640px) {
		.chat-main {
			padding: 0.75rem 0.75rem max(0.75rem, env(safe-area-inset-bottom));
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
