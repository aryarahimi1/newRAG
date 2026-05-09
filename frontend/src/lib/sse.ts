export type SSEFrame = { event: string; data: string };

/** Split buffered SSE stream text into complete `\n\n`-delimited frames; returns remainder buffer. */
export function consumeSSEChunks(buffer: string, chunk: string): { buffer: string; frames: SSEFrame[] } {
	buffer += chunk;
	const parts = buffer.split('\n\n');
	const remainder = parts.pop() ?? '';
	const frames: SSEFrame[] = [];

	for (const part of parts) {
		if (!part.trim()) continue;

		let event = '';
		const dataLines: string[] = [];
		for (const line of part.split('\n')) {
			if (line.startsWith('event: ')) event = line.slice(7).trim();
			else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
		}

		if (event && dataLines.length) {
			frames.push({ event, data: dataLines.join('\n') });
		}
	}

	return { buffer: remainder, frames };
}
