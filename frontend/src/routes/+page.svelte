<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';

	import ChatTopBar from '$lib/components/ChatTopBar.svelte';
	import Composer from '$lib/components/Composer.svelte';
	import DevMetaRow from '$lib/components/DevMetaRow.svelte';
	import MessageThread from '$lib/components/MessageThread.svelte';
	import PipelinePanel from '$lib/components/PipelinePanel.svelte';
	import SessionRail from '$lib/components/SessionRail.svelte';

	import { buildHistory, titleFromQuestion } from '$lib/chat-format';
	import { extractRecallAlerts } from '$lib/recall';
	import { consumeSSEChunks } from '$lib/sse';
	import { applyTheme, resolveInitialTheme, type Theme } from '$lib/theme';
	import type {
		ChatSession,
		ChatTurn,
		CorpusStats,
		PipelineResult,
		StatusEntry
	} from '$lib/types/chat';

	const SESSIONS_STORAGE_KEY = 'medication-reference-chat-sessions-v1';
	const DEV_MODE_STORAGE_KEY = 'medication-reference-dev-mode';
	const RAIL_HEADING_ID = 'rail-session-heading';

	const sampleQuestions = [
		'Can I take ibuprofen with lisinopril for my blood pressure?',
		'Is it safe to combine warfarin and aspirin?',
		"I'm John Smith (john@example.com). Does metformin interact with alcohol?",
		'What happens if I take sertraline and tramadol together?',
		'Can I take omeprazole while on clopidogrel?',
		'Does rifampin reduce the effectiveness of warfarin?'
	];

	let msgSeq = $state(0);
	let chatSeq = $state(0);
	let statusSeq = $state(0);

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
	let sessionStorageReady = $state(false);
	let railCollapsed = $state(true);

	let isMobile = $state(false);
	let controlsOpen = $state(false);

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
				controlsOpen = false;
			} else {
				sessionStorage.setItem(DEV_MODE_STORAGE_KEY, '1');
				devModeEnabled = true;
			}
		} catch {
			devModeEnabled = !devModeEnabled;
			if (!devModeEnabled) controlsOpen = false;
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
	let compactPipelineUi = $state(false);
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

	let flashClearTimer: ReturnType<typeof setTimeout> | undefined;
	let persistTimer: ReturnType<typeof setTimeout> | undefined;
	let mediaCleanup: (() => void) | undefined;

	async function scrollThreadToEnd(behavior: ScrollBehavior = 'smooth') {
		await tick();
		threadEl?.scrollTo({ top: threadEl.scrollHeight, behavior });
	}

	function threadScrollAttach(el: HTMLDivElement) {
		threadEl = el;
		return () => {
			threadEl = undefined;
		};
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

	function flushPersist() {
		persistTimer = undefined;
		if (!browser || !sessionStorageReady) return;
		try {
			localStorage.setItem(
				SESSIONS_STORAGE_KEY,
				JSON.stringify({ sessions, activeChatId, msgSeq, chatSeq, statusSeq })
			);
		} catch {
			/* quota / private mode */
		}
	}

	$effect(() => {
		void sessions;
		void activeChatId;
		void msgSeq;
		void chatSeq;
		void statusSeq;
		void sessionStorageReady;
		if (!browser || !sessionStorageReady) return;
		clearTimeout(persistTimer);
		persistTimer = setTimeout(flushPersist, 250);
		return () => clearTimeout(persistTimer);
	});

	$effect(() => {
		if (!browser) return;
		const lock = isMobile && !railCollapsed;
		document.documentElement.toggleAttribute('data-scroll-lock', lock);
		return () => {
			document.documentElement.removeAttribute('data-scroll-lock');
		};
	});

	onMount(() => {
		if (!browser) return;

		void loadMeta();

		let wasMobile = window.matchMedia('(max-width: 920px)').matches;
		const mqMobile = window.matchMedia('(max-width: 920px)');
		const mqCompact = window.matchMedia('(max-width: 640px)');

		const syncMedia = () => {
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

		syncMedia();
		mqMobile.addEventListener('change', syncMedia);
		mqCompact.addEventListener('change', syncMedia);
		mediaCleanup = () => {
			mqMobile.removeEventListener('change', syncMedia);
			mqCompact.removeEventListener('change', syncMedia);
		};

		if (!mqMobile.matches) {
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
				const next =
					qs ?
						`${window.location.pathname}?${qs}${window.location.hash}`
					:	`${window.location.pathname}${window.location.hash}`;
				void goto(resolve(next as '/' | `/?${string}` | `/#${string}` | `/?${string}#${string}`), {
					replaceState: true,
					keepFocus: true,
					noScroll: true
				});
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
			/* ignore corrupt storage */
		} finally {
			sessionStorageReady = true;
		}

		return () => {
			mediaCleanup?.();
			clearTimeout(flashClearTimer);
			clearTimeout(persistTimer);
			flushPersist();
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
		if (flashClearTimer) clearTimeout(flashClearTimer);
		flashClearTimer = setTimeout(() => {
			lastSentUserId = null;
			flashClearTimer = undefined;
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
					ensureAssistantBubble();
					const recalls = extractRecallAlerts(result);
					updateChat(chatId, (session) => {
						let messages = session.messages;
						if (assistantId !== null) {
							messages = session.messages.map((m) =>
								m.id === assistantId ?
										{
											...m,
											recallAlerts: recalls.length ? recalls : undefined
										}
									:	m
							);
						}
						return {
							...session,
							messages,
							lastResult: result,
							statusLines: session.statusLines.length ?
									session.statusLines
								:	statusEntries(result.status_log ?? []),
							requestError: result.error,
							updatedAt: Date.now()
						};
					});

					if (!streamedContent) {
						ensureAssistantBubble();
						const answer =
							result.generation?.answer ??
							(result.error ?
								`_(generation failed: ${result.error})_`
							:	'_(LLM call skipped - retrieval only)_');
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
				/* malformed stream frames ignored */
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
				const chunk = decoder.decode(value, { stream: true });
				const consumed = consumeSSEChunks(buffer, chunk);
				buffer = consumed.buffer;
				for (const frame of consumed.frames) {
					handleSSEEvent(frame.event, frame.data);
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
	{#if showMobileBackdrop}
		<button
			type="button"
			class="mobile-overlay-backdrop"
			aria-label="Close panel"
			onclick={dismissMobileOverlays}
		></button>
	{/if}
	<SessionRail
		{sessions}
		{activeChatId}
		{railCollapsed}
		{isMobile}
		railHeadingId={RAIL_HEADING_ID}
		onSelectChat={selectChat}
		onNewChat={startNewChat}
		onToggleRail={toggleRailCollapsed}
	/>

	<main class="chat-main" inert={isMobile && !railCollapsed}>
		<div class="main-max">
			<ChatTopBar
				{isMobile}
				{theme}
				hasMessages={hasMessages}
				hasPipelineResult={currentResult !== null}
				{railCollapsed}
				onOpenRail={openMobileSessionRail}
				onToggleTheme={toggleTheme}
				onClearChat={clearActiveChat}
			/>

			{#if devModeEnabled && hasAssistantReply}
				<DevMetaRow {corpus} {apiConfig} />
			{/if}

			{#if currentError}
				<div class="banner error" role="alert">{currentError}</div>
			{/if}

			<MessageThread
				hasMessages={hasMessages}
				{currentMessages}
				{loading}
				{streamingMsgId}
				{streamingPreAnswer}
				{lastSentUserId}
				{latestStatus}
				{sampleQuestions}
				threadAttach={threadScrollAttach}
				onAskSample={send}
			/>

			{#if devModeEnabled && (currentStatus.length || currentResult)}
				<PipelinePanel
					bind:pipelineDetailsOpen
					{currentStatus}
					currentResult={currentResult}
					{loading}
					{compactPipelineUi}
				/>
			{/if}

			<Composer
				bind:input
				{loading}
				{devModeEnabled}
				bind:controlsOpen
				bind:topKRetrieve
				bind:topKRerank
				bind:autoIngest
				bind:skipGeneration
				apiHasKey={apiConfig?.has_openrouter_key}
				{onSubmit}
				onComposerKeydown={handleComposerKeydown}
				{onControlsToggle}
			/>
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

	.banner {
		border-radius: 0.9rem;
		padding: 0.8rem 1rem;
	}

	.banner.error {
		background: var(--error-banner-bg);
		border: 1px solid var(--error-banner-border);
		color: var(--error-banner-fg);
	}

	button.mobile-overlay-backdrop {
		display: none;
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

		.app-shell,
		.app-shell.rail-minimized {
			display: block;
		}

		.chat-main {
			height: auto;
			min-height: 100dvh;
			padding-bottom: max(1.1rem, env(safe-area-inset-bottom));
		}
	}

	@media (max-width: 640px) {
		.chat-main {
			padding: 0.75rem 0.75rem max(0.75rem, env(safe-area-inset-bottom));
		}
	}
</style>
