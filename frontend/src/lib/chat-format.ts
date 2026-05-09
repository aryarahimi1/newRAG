import type { ChatSession, ChatTurn, MetadataValue } from '$lib/types/chat';

export function metadataValue(value: MetadataValue): string {
	if (value === null || value === undefined) return '';
	return String(value);
}

export function titleFromQuestion(question: string): string {
	const clean = question.replace(/\s+/g, ' ').trim();
	return clean.length > 42 ? `${clean.slice(0, 39)}...` : clean || 'New chat';
}

function firstUserContent(session: ChatSession): string | undefined {
	const turn = session.messages.find((m) => m.role === 'user');
	const t = turn?.content?.replace(/\s+/g, ' ').trim();
	return t || undefined;
}

/** Sidebar primary line: always follows the first user question when present. */
export function chatListTitle(session: ChatSession): string {
	const first = firstUserContent(session);
	if (first) return titleFromQuestion(first);
	return 'Empty conversation';
}

/** Sidebar secondary line: status / last activity snippet. */
export function chatListPreview(session: ChatSession): string {
	const msgs = session.messages;
	if (msgs.length === 0) return 'Ask your first question below';
	const last = msgs.at(-1)!;
	if (last.role === 'user' && msgs.length === 1) return 'Awaiting response…';
	const raw = last.content.replace(/\s+/g, ' ').trim() || '…';
	return raw.length > 52 ? `${raw.slice(0, 49)}…` : raw;
}

export function buildHistory(messages: ChatTurn[]): ChatTurn[] {
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

export function formatTime(value: number): string {
	return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(value);
}

export function formatScore(value: number | null | undefined): string {
	if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
	return value.toFixed(3);
}

export function formatMs(value: number): string {
	return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${value.toFixed(0)}ms`;
}
